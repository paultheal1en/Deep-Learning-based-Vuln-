static void sclp_execute(SCLPDevice *sclp, SCCB *sccb, uint32_t code)

{

    SCLPDeviceClass *sclp_c = SCLP_GET_CLASS(sclp);

    SCLPEventFacility *ef = sclp->event_facility;

    SCLPEventFacilityClass *efc = EVENT_FACILITY_GET_CLASS(ef);



    switch (code & SCLP_CMD_CODE_MASK) {

    case SCLP_CMDW_READ_SCP_INFO:

    case SCLP_CMDW_READ_SCP_INFO_FORCED:

        sclp_c->read_SCP_info(sclp, sccb);

        break;

    case SCLP_CMDW_READ_CPU_INFO:

        sclp_c->read_cpu_info(sclp, sccb);

        break;

    case SCLP_READ_STORAGE_ELEMENT_INFO:

        if (code & 0xff00) {

            sclp_c->read_storage_element1_info(sclp, sccb);

        } else {

            sclp_c->read_storage_element0_info(sclp, sccb);

        }

        break;

    case SCLP_ATTACH_STORAGE_ELEMENT:

        sclp_c->attach_storage_element(sclp, sccb, (code & 0xff00) >> 8);

        break;

    case SCLP_ASSIGN_STORAGE:

        sclp_c->assign_storage(sclp, sccb);

        break;

    case SCLP_UNASSIGN_STORAGE:

        sclp_c->unassign_storage(sclp, sccb);

        break;

    case SCLP_CMDW_CONFIGURE_PCI:

        s390_pci_sclp_configure(sccb);

        break;

    case SCLP_CMDW_DECONFIGURE_PCI:

        s390_pci_sclp_deconfigure(sccb);

        break;

    default:

        efc->command_handler(ef, sccb, code);

        break;

    }

}
