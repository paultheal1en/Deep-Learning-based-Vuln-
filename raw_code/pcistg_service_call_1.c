int pcistg_service_call(S390CPU *cpu, uint8_t r1, uint8_t r2)

{

    CPUS390XState *env = &cpu->env;

    uint64_t offset, data;

    S390PCIBusDevice *pbdev;

    MemoryRegion *mr;

    uint8_t len;

    uint32_t fh;

    uint8_t pcias;



    cpu_synchronize_state(CPU(cpu));



    if (env->psw.mask & PSW_MASK_PSTATE) {

        program_interrupt(env, PGM_PRIVILEGED, 4);

        return 0;

    }



    if (r2 & 0x1) {

        program_interrupt(env, PGM_SPECIFICATION, 4);

        return 0;

    }



    fh = env->regs[r2] >> 32;

    pcias = (env->regs[r2] >> 16) & 0xf;

    len = env->regs[r2] & 0xf;

    offset = env->regs[r2 + 1];



    pbdev = s390_pci_find_dev_by_fh(fh);

    if (!pbdev) {

        DPRINTF("pcistg no pci dev\n");

        setcc(cpu, ZPCI_PCI_LS_INVAL_HANDLE);

        return 0;

    }



    switch (pbdev->state) {

    case ZPCI_FS_RESERVED:

    case ZPCI_FS_STANDBY:

    case ZPCI_FS_DISABLED:

    case ZPCI_FS_PERMANENT_ERROR:

        setcc(cpu, ZPCI_PCI_LS_INVAL_HANDLE);

        return 0;

    case ZPCI_FS_ERROR:

        setcc(cpu, ZPCI_PCI_LS_ERR);

        s390_set_status_code(env, r2, ZPCI_PCI_ST_BLOCKED);

        return 0;

    default:

        break;

    }



    data = env->regs[r1];

    if (pcias < 6) {

        if ((8 - (offset & 0x7)) < len) {

            program_interrupt(env, PGM_OPERAND, 4);

            return 0;

        }



        if (trap_msix(pbdev, offset, pcias)) {

            offset = offset - pbdev->msix.table_offset;

            mr = &pbdev->pdev->msix_table_mmio;

            update_msix_table_msg_data(pbdev, offset, &data, len);

        } else {

            mr = pbdev->pdev->io_regions[pcias].memory;

        }



        memory_region_dispatch_write(mr, offset, data, len,

                                     MEMTXATTRS_UNSPECIFIED);

    } else if (pcias == 15) {

        if ((4 - (offset & 0x3)) < len) {

            program_interrupt(env, PGM_OPERAND, 4);

            return 0;

        }

        switch (len) {

        case 1:

            break;

        case 2:

            data = bswap16(data);

            break;

        case 4:

            data = bswap32(data);

            break;

        case 8:

            data = bswap64(data);

            break;

        default:

            program_interrupt(env, PGM_OPERAND, 4);

            return 0;

        }



        pci_host_config_write_common(pbdev->pdev, offset,

                                     pci_config_size(pbdev->pdev),

                                     data, len);

    } else {

        DPRINTF("pcistg invalid space\n");

        setcc(cpu, ZPCI_PCI_LS_ERR);

        s390_set_status_code(env, r2, ZPCI_PCI_ST_INVAL_AS);

        return 0;

    }



    setcc(cpu, ZPCI_PCI_LS_OK);

    return 0;

}
