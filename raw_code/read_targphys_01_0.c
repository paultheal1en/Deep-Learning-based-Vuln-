int read_targphys(int fd, target_phys_addr_t dst_addr, size_t nbytes)

{

    uint8_t buf[4096];

    target_phys_addr_t dst_begin = dst_addr;

    size_t want, did;



    while (nbytes) {

	want = nbytes > sizeof(buf) ? sizeof(buf) : nbytes;

	did = read(fd, buf, want);

	if (did != want) break;



	cpu_physical_memory_write_rom(dst_addr, buf, did);

	dst_addr += did;

	nbytes -= did;

    }

    return dst_addr - dst_begin;

}
