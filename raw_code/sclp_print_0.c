void sclp_print(const char *str)

{

    int len = _strlen(str);

    WriteEventData *sccb = (void*)_sccb;



    sccb->h.length = sizeof(WriteEventData) + len;

    sccb->h.function_code = SCLP_FC_NORMAL_WRITE;

    sccb->ebh.length = sizeof(EventBufferHeader) + len;

    sccb->ebh.type = SCLP_EVENT_ASCII_CONSOLE_DATA;

    sccb->ebh.flags = 0;

    _memcpy(sccb->data, str, len);



    sclp_service_call(SCLP_CMD_WRITE_EVENT_DATA, sccb);

}
