static void mmubooke_create_initial_mapping(CPUPPCState *env)

{

    struct boot_info *bi = env->load_info;

    ppcmas_tlb_t *tlb = booke206_get_tlbm(env, 1, 0, 0);

    hwaddr size, dt_end;

    int ps;



    /* Our initial TLB entry needs to cover everything from 0 to

       the device tree top */

    dt_end = bi->dt_base + bi->dt_size;

    ps = booke206_page_size_to_tlb(dt_end) + 1;

    if (ps & 1) {

        /* e500v2 can only do even TLB size bits */

        ps++;

    }

    size = (ps << MAS1_TSIZE_SHIFT);

    tlb->mas1 = MAS1_VALID | size;

    tlb->mas2 = 0;

    tlb->mas7_3 = 0;

    tlb->mas7_3 |= MAS3_UR | MAS3_UW | MAS3_UX | MAS3_SR | MAS3_SW | MAS3_SX;



    env->tlb_dirty = true;

}
