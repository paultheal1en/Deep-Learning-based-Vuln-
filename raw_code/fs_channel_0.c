static inline int fs_channel(target_phys_addr_t addr)

{

	/* Every channel has a 0x2000 ctrl register map.  */

	return addr >> 13;

}
