static void gt64120_isd_mapping(GT64120State *s)

{

    target_phys_addr_t start = s->regs[GT_ISD] << 21;

    target_phys_addr_t length = 0x1000;



    if (s->ISD_length) {

        memory_region_del_subregion(get_system_memory(), &s->ISD_mem);

    }

    check_reserved_space(&start, &length);

    length = 0x1000;

    /* Map new address */

    DPRINTF("ISD: "TARGET_FMT_plx"@"TARGET_FMT_plx

        " -> "TARGET_FMT_plx"@"TARGET_FMT_plx"\n",

        s->ISD_length, s->ISD_start, length, start);

    s->ISD_start = start;

    s->ISD_length = length;

    memory_region_add_subregion(get_system_memory(), s->ISD_start, &s->ISD_mem);

}
