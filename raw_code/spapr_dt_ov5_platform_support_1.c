static void spapr_dt_ov5_platform_support(void *fdt, int chosen)

{

    PowerPCCPU *first_ppc_cpu = POWERPC_CPU(first_cpu);



    char val[2 * 4] = {

        23, 0x00, /* Xive mode: 0 = legacy (as in ISA 2.7), 1 = Exploitation */

        24, 0x00, /* Hash/Radix, filled in below. */

        25, 0x00, /* Hash options: Segment Tables == no, GTSE == no. */

        26, 0x40, /* Radix options: GTSE == yes. */

    };



    if (kvm_enabled()) {

        if (kvmppc_has_cap_mmu_radix() && kvmppc_has_cap_mmu_hash_v3()) {

            val[3] = 0x80; /* OV5_MMU_BOTH */

        } else if (kvmppc_has_cap_mmu_radix()) {

            val[3] = 0x40; /* OV5_MMU_RADIX_300 */

        } else {

            val[3] = 0x00; /* Hash */

        }

    } else {

        if (first_ppc_cpu->env.mmu_model & POWERPC_MMU_V3) {

            /* V3 MMU supports both hash and radix (with dynamic switching) */

            val[3] = 0xC0;

        } else {

            /* Otherwise we can only do hash */

            val[3] = 0x00;

        }

    }

    _FDT(fdt_setprop(fdt, chosen, "ibm,arch-vec-5-platform-support",

                     val, sizeof(val)));

}
