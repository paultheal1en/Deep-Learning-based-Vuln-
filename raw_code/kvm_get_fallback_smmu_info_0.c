static void kvm_get_fallback_smmu_info(PowerPCCPU *cpu,

                                       struct kvm_ppc_smmu_info *info)

{

    CPUPPCState *env = &cpu->env;

    CPUState *cs = CPU(cpu);



    memset(info, 0, sizeof(*info));



    /* We don't have the new KVM_PPC_GET_SMMU_INFO ioctl, so

     * need to "guess" what the supported page sizes are.

     *

     * For that to work we make a few assumptions:

     *

     * - If KVM_CAP_PPC_GET_PVINFO is supported we are running "PR"

     *   KVM which only supports 4K and 16M pages, but supports them

     *   regardless of the backing store characteritics. We also don't

     *   support 1T segments.

     *

     *   This is safe as if HV KVM ever supports that capability or PR

     *   KVM grows supports for more page/segment sizes, those versions

     *   will have implemented KVM_CAP_PPC_GET_SMMU_INFO and thus we

     *   will not hit this fallback

     *

     * - Else we are running HV KVM. This means we only support page

     *   sizes that fit in the backing store. Additionally we only

     *   advertize 64K pages if the processor is ARCH 2.06 and we assume

     *   P7 encodings for the SLB and hash table. Here too, we assume

     *   support for any newer processor will mean a kernel that

     *   implements KVM_CAP_PPC_GET_SMMU_INFO and thus doesn't hit

     *   this fallback.

     */

    if (kvm_check_extension(cs->kvm_state, KVM_CAP_PPC_GET_PVINFO)) {

        /* No flags */

        info->flags = 0;

        info->slb_size = 64;



        /* Standard 4k base page size segment */

        info->sps[0].page_shift = 12;

        info->sps[0].slb_enc = 0;

        info->sps[0].enc[0].page_shift = 12;

        info->sps[0].enc[0].pte_enc = 0;



        /* Standard 16M large page size segment */

        info->sps[1].page_shift = 24;

        info->sps[1].slb_enc = SLB_VSID_L;

        info->sps[1].enc[0].page_shift = 24;

        info->sps[1].enc[0].pte_enc = 0;

    } else {

        int i = 0;



        /* HV KVM has backing store size restrictions */

        info->flags = KVM_PPC_PAGE_SIZES_REAL;



        if (env->mmu_model & POWERPC_MMU_1TSEG) {

            info->flags |= KVM_PPC_1T_SEGMENTS;

        }



        if (env->mmu_model == POWERPC_MMU_2_06 ||

            env->mmu_model == POWERPC_MMU_2_07) {

            info->slb_size = 32;

        } else {

            info->slb_size = 64;

        }



        /* Standard 4k base page size segment */

        info->sps[i].page_shift = 12;

        info->sps[i].slb_enc = 0;

        info->sps[i].enc[0].page_shift = 12;

        info->sps[i].enc[0].pte_enc = 0;

        i++;



        /* 64K on MMU 2.06 and later */

        if (env->mmu_model == POWERPC_MMU_2_06 ||

            env->mmu_model == POWERPC_MMU_2_07) {

            info->sps[i].page_shift = 16;

            info->sps[i].slb_enc = 0x110;

            info->sps[i].enc[0].page_shift = 16;

            info->sps[i].enc[0].pte_enc = 1;

            i++;

        }



        /* Standard 16M large page size segment */

        info->sps[i].page_shift = 24;

        info->sps[i].slb_enc = SLB_VSID_L;

        info->sps[i].enc[0].page_shift = 24;

        info->sps[i].enc[0].pte_enc = 0;

    }

}
