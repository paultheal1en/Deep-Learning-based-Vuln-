static int cris_mmu_segmented_addr(int seg, uint32_t rw_mm_cfg)

{

	return (1 << seg) & rw_mm_cfg;

}
