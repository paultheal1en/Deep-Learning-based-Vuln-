static void sclp_set_write_mask(void)

{

    WriteEventMask *sccb = (void*)_sccb;



    sccb->h.length = sizeof(WriteEventMask);

    sccb->mask_length = sizeof(unsigned int);

    sccb->receive_mask = SCLP_EVENT_MASK_MSG_ASCII;

    sccb->cp_receive_mask = SCLP_EVENT_MASK_MSG_ASCII;

    sccb->send_mask = SCLP_EVENT_MASK_MSG_ASCII;

    sccb->cp_send_mask = SCLP_EVENT_MASK_MSG_ASCII;



    sclp_service_call(SCLP_CMD_WRITE_EVENT_MASK, sccb);

}
