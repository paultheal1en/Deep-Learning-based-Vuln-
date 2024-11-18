static void tlb_flush_by_mmuidx_async_work(CPUState *cpu, run_on_cpu_data data)

{

    CPUArchState *env = cpu->env_ptr;

    unsigned long mmu_idx_bitmask = data.host_int;

    int mmu_idx;



    assert_cpu_is_self(cpu);



    tb_lock();



    tlb_debug("start: mmu_idx:0x%04lx\n", mmu_idx_bitmask);



    for (mmu_idx = 0; mmu_idx < NB_MMU_MODES; mmu_idx++) {



        if (test_bit(mmu_idx, &mmu_idx_bitmask)) {

            tlb_debug("%d\n", mmu_idx);



            memset(env->tlb_table[mmu_idx], -1, sizeof(env->tlb_table[0]));

            memset(env->tlb_v_table[mmu_idx], -1, sizeof(env->tlb_v_table[0]));

        }

    }



    memset(cpu->tb_jmp_cache, 0, sizeof(cpu->tb_jmp_cache));



    tlb_debug("done\n");



    tb_unlock();

}
