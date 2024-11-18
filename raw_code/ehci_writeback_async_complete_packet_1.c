static void ehci_writeback_async_complete_packet(EHCIPacket *p)
{
    EHCIQueue *q = p->queue;
    int state;
    state = ehci_get_state(q->ehci, q->async);
    ehci_state_executing(q);
    ehci_state_writeback(q); /* Frees the packet! */
    if (!(q->qh.token & QTD_TOKEN_HALT)) {
        ehci_state_advqueue(q);
    ehci_set_state(q->ehci, q->async, state);