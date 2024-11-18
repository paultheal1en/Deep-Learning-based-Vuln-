void cpu_ppc_set_papr(PowerPCCPU *cpu)

{

    CPUPPCState *env = &cpu->env;




    /* PAPR always has exception vectors in RAM not ROM. To ensure this,

     * MSR[IP] should never be set.

     *

     * We also disallow setting of MSR_HV

     */

    env->msr_mask &= ~((1ull << MSR_EP) | MSR_HVB);



    /* Set a full AMOR so guest can use the AMR as it sees fit */

    env->spr[SPR_AMOR] = amor->default_value = 0xffffffffffffffffull;



    /* Tell KVM that we're in PAPR mode */

    if (kvm_enabled()) {

        kvmppc_set_papr(cpu);

    }

}