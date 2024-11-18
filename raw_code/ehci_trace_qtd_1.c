static void ehci_trace_qtd(EHCIQueue *q, target_phys_addr_t addr, EHCIqtd *qtd)

{

    trace_usb_ehci_qtd(q, addr, qtd->next, qtd->altnext,

                       get_field(qtd->token, QTD_TOKEN_TBYTES),

                       get_field(qtd->token, QTD_TOKEN_CPAGE),

                       get_field(qtd->token, QTD_TOKEN_CERR),

                       get_field(qtd->token, QTD_TOKEN_PID),

                       (bool)(qtd->token & QTD_TOKEN_IOC),

                       (bool)(qtd->token & QTD_TOKEN_ACTIVE),

                       (bool)(qtd->token & QTD_TOKEN_HALT),

                       (bool)(qtd->token & QTD_TOKEN_BABBLE),

                       (bool)(qtd->token & QTD_TOKEN_XACTERR));

}
