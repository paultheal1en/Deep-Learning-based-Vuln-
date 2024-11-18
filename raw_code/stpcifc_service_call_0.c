int stpcifc_service_call(S390CPU *cpu, uint8_t r1, uint64_t fiba, uint8_t ar)

{

    CPUS390XState *env = &cpu->env;

    uint32_t fh;

    ZpciFib fib;

    S390PCIBusDevice *pbdev;

    uint32_t data;

    uint64_t cc = ZPCI_PCI_LS_OK;



    if (env->psw.mask & PSW_MASK_PSTATE) {

        program_interrupt(env, PGM_PRIVILEGED, 6);

        return 0;

    }



    fh = env->regs[r1] >> 32;



    if (fiba & 0x7) {

        program_interrupt(env, PGM_SPECIFICATION, 6);

        return 0;

    }



    pbdev = s390_pci_find_dev_by_fh(fh);

    if (!pbdev) {

        setcc(cpu, ZPCI_PCI_LS_INVAL_HANDLE);

        return 0;

    }



    memset(&fib, 0, sizeof(fib));

    stq_p(&fib.pba, pbdev->pba);

    stq_p(&fib.pal, pbdev->pal);

    stq_p(&fib.iota, pbdev->g_iota);

    stq_p(&fib.aibv, pbdev->routes.adapter.ind_addr);

    stq_p(&fib.aisb, pbdev->routes.adapter.summary_addr);

    stq_p(&fib.fmb_addr, pbdev->fmb_addr);



    data = ((uint32_t)pbdev->isc << 28) | ((uint32_t)pbdev->noi << 16) |

           ((uint32_t)pbdev->routes.adapter.ind_offset << 8) |

           ((uint32_t)pbdev->sum << 7) | pbdev->routes.adapter.summary_offset;

    stl_p(&fib.data, data);



    if (pbdev->fh & FH_MASK_ENABLE) {

        fib.fc |= 0x80;

    }



    if (pbdev->error_state) {

        fib.fc |= 0x40;

    }



    if (pbdev->lgstg_blocked) {

        fib.fc |= 0x20;

    }



    if (pbdev->g_iota) {

        fib.fc |= 0x10;

    }



    if (s390_cpu_virt_mem_write(cpu, fiba, ar, (uint8_t *)&fib, sizeof(fib))) {

        return 0;

    }



    setcc(cpu, cc);

    return 0;

}
