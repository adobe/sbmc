# This Makefile defines rules to build and run a Docker environment for easier
# reproducibility, as well as commands demonstrating how to use the code
# (commands starting with `demo/`).

OUTPUT?=$(shell pwd)/output
DATA?=$(shell pwd)/data
PBRT?=_extras/rendernet_pbrt/src/bin/pbrt
OBJ2PBRT?=_extras/rendernet_pbrt/src/bin/obj2pbrt

# Data server
REMOTE=http://ephesia/~mgharbi/sbmc

# Path to comparison code
SEN2011?=_extras/comparisons/methods/2011_sen_rpf
ROUSSELLE2012?=_extras/comparisons/methods/2012_rousselle_nlm
KALANTARI2015?=_extras/comparisons/methods/2015_kalantari_lbf
BITTERLI2016?=_extras/comparisons/methods/nfor_fromdocker

# Install the required extension for CUDA on Docker
nvidia_docker:
	./scripts/install_nvidia_docker.sh

# To facilitate environment setup, build and use this dockerfile
# !! Requires NVIDIA's docker extension !!
docker_build:
	@docker build -f dockerfiles/cuda-sbmc.dockerfile -t sbmc_cuda .

# This target launches a fully configured docker instance,
# mounts $(OUTPUT) as read-write volume and $(DATA) as readonly for persisten I/O.
# Once logged into the docker instance, you can run any of the `make demo/*`
# commands.
docker_run:  docker_build
	@docker run --gpus all --name sbmc_cuda_app --rm \
		-v $(OUTPUT):/sbmc_app/output \
		-v $(DATA):/sbmc_app/data:ro \
		--ipc=host \
		-p 2001:2001 \
		-it sbmc_cuda

clean:
	rm -rf dist .pytest_cache sbmc.egg-info build sbmc/halide_ops.*.so \
		$(OUTPUT)/demo*

.PHONY: demo/render_bins demo/render_reference demo/generate_scenes \
	demo/visualize demo/denoiser demo/train demo/train_kpcn \
	demo/render_samples server clean nvidia_docker


# -----------------------------------------------------------------------------

# The rest of this Makefiles demonstrates how to use the SBMC API and entry
# scripts for common tasks.

$(DATA):
	mkdir -p $(DATA)

# Download the data needed for the demo
demo/data: $(DATA)/dummy.txt

$(DATA)/dummy.txt: $(DATA)
	@python scripts/download.py $(REMOTE)/dummy.zip $(DATA)/dummy.zip
	cd $(DATA) && unzip dummy.zip
	rm $(DATA)/dummy.zip

pretrained_models: $(DATA)/pretrained_models/gharbi2019_sbmc/final.pth

$(DATA)/pretrained_models/gharbi2019_sbmc/final.pth: $(DATA)
	@python scripts/download.py $(REMOTE)/pretrained_models.zip $(DATA)/pretrained_models.zip
	cd $(DATA) && unzip pretrained_models.zip
	rm $(DATA)/pretrained_models.zip

demo/render_samples: $(OUTPUT)/demo/test_samples/0000_0000.bin

# This demonstrates how we render .bin sample files for a test scene
$(OUTPUT)/demo/test_samples/0000_0000.bin: demo/data
	@python scripts/render_samples.py $(PBRT) \
		$(DATA)/demo/scenes/GITestSynthesizer_01/scene.pbrt \
		$(OUTPUT)/demo/test_samples \
		--tmp_dir $(OUTPUT)/tmp --height 128 --width 128 --spp 4 \
		--gt_spp 1

# This demonstrates how we render .bin sample files for a training dataset 
# using our random scene generator
demo/generate_scenes: demo/data
	@python scripts/generate_training_data.py $(PBRT) \
		$(OBJ2PBRT) \
		$(DATA)/demo/scenegen_assets $(OUTPUT)/demo/training_scenes --count 2 \
		--spp 1 --gt_spp 4 --height 128 --width 128

# This shows how to use the visualization helper script to inspect the sample
# .bin files
demo/visualize:
	@python scripts/visualize_dataset.py $(OUTPUT)/demo/training_scenes \
		$(OUTPUT)/demo/dataviz --spp 1

# This demonstrates how to run pretrained models on .bin test scenes
demo/denoise: demo/render_samples pretrained_models
	@python scripts/denoise.py \
		--input $(OUTPUT)/demo/test_samples \
		--output $(OUTPUT)/demo/ours_4spp.exr \
		--spp 4 \
		--checkpoint $(DATA)/pretrained_models/gharbi2019_sbmc
	@python scripts/denoise.py \
		--input $(OUTPUT)/demo/test_samples \
		--output $(OUTPUT)/demo/bako2017_4spp.exr \
		--spp 4 \
		--checkpoint $(DATA)/pretrained_models/bako2017_finetuned


# This demonstrates how we render a .exr reference image for a test scene
demo/render_reference: demo/data
	@python scripts/render_exr.py $(PBRT) \
		$(DATA)/demo/scenes/GITestSynthesizer_01/scene.pbrt \
		$(OUTPUT)/demo/comparisons/reference/GITestSynthesizer_01.exr \
		--tmp_dir $(OUTPUT)/tmp --height 128 --width 128 --spp 4

# This demonstrates how we render .exr images for the comparison denoisers
demo/comparisons: demo/render_samples pretrained_models demo/data
	@python scripts/denoise.py \
		--input $(OUTPUT)/demo/test_samples \
		--output $(OUTPUT)/demo/comparisons/2017_bako_kpcn_finetuned/GITestSynthesizer_01.exr \
		--spp 4 \
		--checkpoint $(DATA)/pretrained_models/bako2017_finetuned
	@python scripts/denoise.py \
		--input $(OUTPUT)/demo/test_samples \
		--output $(OUTPUT)/demo/comparisons/ours/GITestSynthesizer_01.exr \
		--spp 4 \
		--checkpoint $(DATA)/pretrained_models/gharbi2019_sbmc
	@python scripts/render_exr.py $(SEN2011)/bin/pbrt \
		$(DATA)/demo/scenes/GITestSynthesizer_01/scene.pbrt \
		$(OUTPUT)/demo/comparisons/2011_sen_rpf/GITestSynthesizer_01.exr \
		--tmp_dir $(OUTPUT)/tmp --height 128 --width 128 --spp 4
	@python scripts/render_exr.py $(ROUSSELLE2012)/bin/pbrt \
		$(DATA)/demo/scenes/GITestSynthesizer_01/scene.pbrt \
		$(OUTPUT)/demo/comparisons/2012_rousselle_nlm/GITestSynthesizer_01.exr \
		--tmp_dir $(OUTPUT)/tmp --height 128 --width 128 --spp 4
	@python scripts/render_exr.py $(KALANTARI2015)/bin/pbrt \
		$(DATA)/demo/scenes/GITestSynthesizer_01/scene.pbrt \
		$(OUTPUT)/demo/comparisons/2015_kalantari_lbf/GITestSynthesizer_01.exr \
		--tmp_dir $(OUTPUT)/tmp --height 128 --width 128 --spp 4 \
		--kalantari2015_data $(KALANTARI2015)/pretrained/Weights.dat \
		$(KALANTARI2015)/pretrained/FeatureNorm.dat
	@python scripts/denoise_nfor.py $(BITTERLI2016)/build/denoiser \
		$(OUTPUT)/demo/test_samples \
		$(OUTPUT)/demo/comparisons/2016_bitterli_nfor/GITestSynthesizer_01.exr \
		--tmp_dir $(OUTPUT)/tmp --spp 4

# # This demonstrates how to train a new model
# demo/train: server demo/render_samples
# 	@python scripts/train.py \
# 		--checkpoint_dir $(OUTPUT)/demo/training \
# 		--data $(DATA)/demo/bins/list.txt \
# 		--env sbmc_ours --port 2001 --bs 1 \
# 		--spp 2
#
# # This demonstrates how to train a baseline model (from [Bako 2017])
# demo/train_kpcn: server
# 	@python scripts/train.py \
# 		--checkpoint_dir $(OUTPUT)/demo/training_kpcn \
# 		--data $(DATA)/demo/bins/list.txt \
# 		--constant_spp --env sbmc_kpcn --port 2001 --bs 1 \
# 		--kpcn_mode \
# 		--spp 2

# Launches a Visdom server to monitor the training
server:
	@python -m visdom.server -p 2001 &

demo/eval:
	@python scripts/compute_metrics.py data/renderings/ref output/eval.csv \
		--methods data/eval_methods.txt \
		--scenes data/eval_scenes.txt
