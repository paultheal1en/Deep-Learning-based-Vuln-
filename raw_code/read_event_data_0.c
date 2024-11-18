static void read_event_data(SCLPEventFacility *ef, SCCB *sccb)

{

    unsigned int sclp_active_selection_mask;

    unsigned int sclp_cp_receive_mask;



    ReadEventData *red = (ReadEventData *) sccb;



    if (be16_to_cpu(sccb->h.length) != SCCB_SIZE) {

        sccb->h.response_code = cpu_to_be16(SCLP_RC_INSUFFICIENT_SCCB_LENGTH);

        goto out;

    }



    sclp_cp_receive_mask = ef->receive_mask;



    /* get active selection mask */

    switch (sccb->h.function_code) {

    case SCLP_UNCONDITIONAL_READ:

        sclp_active_selection_mask = sclp_cp_receive_mask;

        break;

    case SCLP_SELECTIVE_READ:

        if (!(sclp_cp_receive_mask & be32_to_cpu(red->mask))) {

            sccb->h.response_code =

                    cpu_to_be16(SCLP_RC_INVALID_SELECTION_MASK);

            goto out;

        }

        sclp_active_selection_mask = be32_to_cpu(red->mask);

        break;

    default:

        sccb->h.response_code = cpu_to_be16(SCLP_RC_INVALID_FUNCTION);

        goto out;

    }

    sccb->h.response_code = cpu_to_be16(

            handle_sccb_read_events(ef, sccb, sclp_active_selection_mask));



out:

    return;

}
