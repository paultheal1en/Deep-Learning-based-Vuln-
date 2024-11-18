static void channel_load_g(struct fs_dma_ctrl *ctrl, int c)

{

	target_phys_addr_t addr = channel_reg(ctrl, c, RW_GROUP);



	/* Load and decode. FIXME: handle endianness.  */

	cpu_physical_memory_read (addr, 

				  (void *) &ctrl->channels[c].current_g, 

				  sizeof ctrl->channels[c].current_g);

}
