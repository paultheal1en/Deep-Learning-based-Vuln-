void op_flush_icache_range(void) {

    CALL_FROM_TB2(tlb_flush_page, env, T0 + T1);

    RETURN();

}
