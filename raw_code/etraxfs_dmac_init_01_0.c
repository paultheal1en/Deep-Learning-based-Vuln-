void *etraxfs_dmac_init(target_phys_addr_t base, int nr_channels)

{

	struct fs_dma_ctrl *ctrl = NULL;



	ctrl = g_malloc0(sizeof *ctrl);



        ctrl->bh = qemu_bh_new(DMA_run, ctrl);



	ctrl->nr_channels = nr_channels;

	ctrl->channels = g_malloc0(sizeof ctrl->channels[0] * nr_channels);



	memory_region_init_io(&ctrl->mmio, &dma_ops, ctrl, "etraxfs-dma",

			      nr_channels * 0x2000);

	memory_region_add_subregion(get_system_memory(), base, &ctrl->mmio);



	return ctrl;

}
