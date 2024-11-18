static uint64_t megasas_mmio_read(void *opaque, target_phys_addr_t addr,

                                  unsigned size)

{

    MegasasState *s = opaque;

    uint32_t retval = 0;



    switch (addr) {

    case MFI_IDB:

        retval = 0;

        break;

    case MFI_OMSG0:

    case MFI_OSP0:

        retval = (megasas_use_msix(s) ? MFI_FWSTATE_MSIX_SUPPORTED : 0) |

            (s->fw_state & MFI_FWSTATE_MASK) |

            ((s->fw_sge & 0xff) << 16) |

            (s->fw_cmds & 0xFFFF);

        break;

    case MFI_OSTS:

        if (megasas_intr_enabled(s) && s->doorbell) {

            retval = MFI_1078_RM | 1;

        }

        break;

    case MFI_OMSK:

        retval = s->intr_mask;

        break;

    case MFI_ODCR0:

        retval = s->doorbell;

        break;

    default:

        trace_megasas_mmio_invalid_readl(addr);

        break;

    }

    trace_megasas_mmio_readl(addr, retval);

    return retval;

}
