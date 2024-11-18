int find_utlb_entry(CPUState * env, target_ulong address, int use_asid)

{

    uint8_t urb, urc;



    /* Increment URC */

    urb = ((env->mmucr) >> 18) & 0x3f;

    urc = ((env->mmucr) >> 10) & 0x3f;

    urc++;

    if (urc == urb || urc == UTLB_SIZE - 1)

	urc = 0;

    env->mmucr = (env->mmucr & 0xffff03ff) | (urc << 10);



    /* Return entry */

    return find_tlb_entry(env, address, env->utlb, UTLB_SIZE, use_asid);

}
