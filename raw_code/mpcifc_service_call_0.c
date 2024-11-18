int mpcifc_service_call(S390CPU *cpu, uint8_t r1, uint64_t fiba, uint8_t ar)

{

    CPUS390XState *env = &cpu->env;

    uint8_t oc, dmaas;

    uint32_t fh;

    ZpciFib fib;

    S390PCIBusDevice *pbdev;

    uint64_t cc = ZPCI_PCI_LS_OK;



    if (env->psw.mask & PSW_MASK_PSTATE) {

        program_interrupt(env, PGM_PRIVILEGED, 6);

        return 0;

    }



    oc = env->regs[r1] & 0xff;

    dmaas = (env->regs[r1] >> 16) & 0xff;

    fh = env->regs[r1] >> 32;



    if (fiba & 0x7) {

        program_interrupt(env, PGM_SPECIFICATION, 6);

        return 0;

    }



    pbdev = s390_pci_find_dev_by_fh(fh);

    if (!pbdev || !(pbdev->fh & FH_MASK_ENABLE)) {

        DPRINTF("mpcifc no pci dev fh 0x%x\n", fh);

        setcc(cpu, ZPCI_PCI_LS_INVAL_HANDLE);

        return 0;

    }



    if (s390_cpu_virt_mem_read(cpu, fiba, ar, (uint8_t *)&fib, sizeof(fib))) {

        return 0;

    }



    if (fib.fmt != 0) {

        program_interrupt(env, PGM_OPERAND, 6);

        return 0;

    }



    switch (oc) {

    case ZPCI_MOD_FC_REG_INT:

        if (pbdev->summary_ind) {

            cc = ZPCI_PCI_LS_ERR;

            s390_set_status_code(env, r1, ZPCI_MOD_ST_SEQUENCE);

        } else if (reg_irqs(env, pbdev, fib)) {

            cc = ZPCI_PCI_LS_ERR;

            s390_set_status_code(env, r1, ZPCI_MOD_ST_RES_NOT_AVAIL);

        }

        break;

    case ZPCI_MOD_FC_DEREG_INT:

        if (!pbdev->summary_ind) {

            cc = ZPCI_PCI_LS_ERR;

            s390_set_status_code(env, r1, ZPCI_MOD_ST_SEQUENCE);

        } else {

            pci_dereg_irqs(pbdev);

        }

        break;

    case ZPCI_MOD_FC_REG_IOAT:

        if (dmaas != 0) {

            cc = ZPCI_PCI_LS_ERR;

            s390_set_status_code(env, r1, ZPCI_MOD_ST_DMAAS_INVAL);

        } else if (pbdev->iommu_enabled) {

            cc = ZPCI_PCI_LS_ERR;

            s390_set_status_code(env, r1, ZPCI_MOD_ST_SEQUENCE);

        } else if (reg_ioat(env, pbdev, fib)) {

            cc = ZPCI_PCI_LS_ERR;

            s390_set_status_code(env, r1, ZPCI_MOD_ST_INSUF_RES);

        }

        break;

    case ZPCI_MOD_FC_DEREG_IOAT:

        if (dmaas != 0) {

            cc = ZPCI_PCI_LS_ERR;

            s390_set_status_code(env, r1, ZPCI_MOD_ST_DMAAS_INVAL);

        } else if (!pbdev->iommu_enabled) {

            cc = ZPCI_PCI_LS_ERR;

            s390_set_status_code(env, r1, ZPCI_MOD_ST_SEQUENCE);

        } else {

            pci_dereg_ioat(pbdev);

        }

        break;

    case ZPCI_MOD_FC_REREG_IOAT:

        if (dmaas != 0) {

            cc = ZPCI_PCI_LS_ERR;

            s390_set_status_code(env, r1, ZPCI_MOD_ST_DMAAS_INVAL);

        } else if (!pbdev->iommu_enabled) {

            cc = ZPCI_PCI_LS_ERR;

            s390_set_status_code(env, r1, ZPCI_MOD_ST_SEQUENCE);

        } else {

            pci_dereg_ioat(pbdev);

            if (reg_ioat(env, pbdev, fib)) {

                cc = ZPCI_PCI_LS_ERR;

                s390_set_status_code(env, r1, ZPCI_MOD_ST_INSUF_RES);

            }

        }

        break;

    case ZPCI_MOD_FC_RESET_ERROR:

        pbdev->error_state = false;

        pbdev->lgstg_blocked = false;

        break;

    case ZPCI_MOD_FC_RESET_BLOCK:

        pbdev->lgstg_blocked = false;

        break;

    case ZPCI_MOD_FC_SET_MEASURE:

        pbdev->fmb_addr = ldq_p(&fib.fmb_addr);

        break;

    default:

        program_interrupt(&cpu->env, PGM_OPERAND, 6);

        cc = ZPCI_PCI_LS_ERR;

    }



    setcc(cpu, cc);

    return 0;

}
