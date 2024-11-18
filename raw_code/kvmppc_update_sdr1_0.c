void kvmppc_update_sdr1(target_ulong sdr1)

{

    CPUState *cs;



    CPU_FOREACH(cs) {

        run_on_cpu(cs, kvmppc_pivot_hpt_cpu, RUN_ON_CPU_TARGET_PTR(sdr1));

    }

}
