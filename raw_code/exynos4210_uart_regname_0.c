static const char *exynos4210_uart_regname(target_phys_addr_t  offset)

{



    int regs_number = sizeof(exynos4210_uart_regs) / sizeof(Exynos4210UartReg);

    int i;



    for (i = 0; i < regs_number; i++) {

        if (offset == exynos4210_uart_regs[i].offset) {

            return exynos4210_uart_regs[i].name;

        }

    }



    return NULL;

}
