static void kvm_fixup_page_sizes(PowerPCCPU *cpu)

{

    static struct kvm_ppc_smmu_info smmu_info;

    static bool has_smmu_info;

    CPUPPCState *env = &cpu->env;

    long rampagesize;

    int iq, ik, jq, jk;



    /* We only handle page sizes for 64-bit server guests for now */

    if (!(env->mmu_model & POWERPC_MMU_64)) {

        return;

    }



    /* Collect MMU info from kernel if not already */

    if (!has_smmu_info) {

        kvm_get_smmu_info(cpu, &smmu_info);

        has_smmu_info = true;

    }



    rampagesize = getrampagesize();



    /* Convert to QEMU form */

    memset(&env->sps, 0, sizeof(env->sps));



    for (ik = iq = 0; ik < KVM_PPC_PAGE_SIZES_MAX_SZ; ik++) {

        struct ppc_one_seg_page_size *qsps = &env->sps.sps[iq];

        struct kvm_ppc_one_seg_page_size *ksps = &smmu_info.sps[ik];



        if (!kvm_valid_page_size(smmu_info.flags, rampagesize,

                                 ksps->page_shift)) {

            continue;

        }

        qsps->page_shift = ksps->page_shift;

        qsps->slb_enc = ksps->slb_enc;

        for (jk = jq = 0; jk < KVM_PPC_PAGE_SIZES_MAX_SZ; jk++) {

            if (!kvm_valid_page_size(smmu_info.flags, rampagesize,

                                     ksps->enc[jk].page_shift)) {

                continue;

            }

            qsps->enc[jq].page_shift = ksps->enc[jk].page_shift;

            qsps->enc[jq].pte_enc = ksps->enc[jk].pte_enc;

            if (++jq >= PPC_PAGE_SIZES_MAX_SZ) {

                break;

            }

        }

        if (++iq >= PPC_PAGE_SIZES_MAX_SZ) {

            break;

        }

    }

    env->slb_nr = smmu_info.slb_size;

    if (smmu_info.flags & KVM_PPC_1T_SEGMENTS) {

        env->mmu_model |= POWERPC_MMU_1TSEG;

    } else {

        env->mmu_model &= ~POWERPC_MMU_1TSEG;

    }

}
