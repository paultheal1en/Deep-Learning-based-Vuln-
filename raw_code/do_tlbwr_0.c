void do_tlbwr (void)

{

    int r = cpu_mips_get_random(env);



    invalidate_tlb(r, 1);

    fill_tlb(r);

}
