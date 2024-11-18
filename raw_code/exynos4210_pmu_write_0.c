static void exynos4210_pmu_write(void *opaque, target_phys_addr_t offset,

                                 uint64_t val, unsigned size)

{

    Exynos4210PmuState *s = (Exynos4210PmuState *)opaque;

    unsigned i;

    const Exynos4210PmuReg *reg_p = exynos4210_pmu_regs;



    for (i = 0; i < PMU_NUM_OF_REGISTERS; i++) {

        if (reg_p->offset == offset) {

            PRINT_DEBUG_EXTEND("%s <0x%04x> <- 0x%04x\n", reg_p->name,

                    (uint32_t)offset, (uint32_t)val);

            s->reg[i] = val;

            return;

        }

        reg_p++;

    }

    PRINT_DEBUG("QEMU PMU ERROR: bad write offset 0x%04x\n", (uint32_t)offset);

}
