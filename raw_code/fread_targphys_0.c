int fread_targphys(target_phys_addr_t dst_addr, size_t nbytes, FILE *f)

{

    uint8_t buf[4096];

    target_phys_addr_t dst_begin = dst_addr;

    size_t want, did;



    while (nbytes) {

	want = nbytes > sizeof(buf) ? sizeof(buf) : nbytes;

	did = fread(buf, 1, want, f);



	cpu_physical_memory_write_rom(dst_addr, buf, did);

	dst_addr += did;

	nbytes -= did;

	if (did != want)

	    break;

    }

    return dst_addr - dst_begin;

}
