static inline int get_dwords(uint32_t addr, uint32_t *buf, int num)

{

    int i;



    for(i = 0; i < num; i++, buf++, addr += sizeof(*buf)) {

        cpu_physical_memory_rw(addr,(uint8_t *)buf, sizeof(*buf), 0);

        *buf = le32_to_cpu(*buf);

    }



    return 1;

}
