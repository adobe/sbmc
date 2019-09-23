scatter_v_gather_figure:
	rm -rf local_output/figures/scatter_v_gather/*.png
	python bin/figures/scatter_v_gather.py local_data/test_images/color.jpg \
		--output local_output/figures/scatter_v_gather --outliers --spp 1

scatter_v_gather_quick:
	rm -rf local_output/figures/scatter_v_gather/*.png
	python bin/figures/scatter_v_gather.py local_data/test_images/color.jpg \
		--output local_output/figures/scatter_v_gather_quick --outliers \
		--spp 2 --nsteps 10
