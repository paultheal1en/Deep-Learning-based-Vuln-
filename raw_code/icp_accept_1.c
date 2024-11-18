static uint32_t icp_accept(struct icp_server_state *ss)

{

    uint32_t xirr = ss->xirr;



    qemu_irq_lower(ss->output);

    ss->xirr = ss->pending_priority << 24;




    trace_xics_icp_accept(xirr, ss->xirr);



    return xirr;

}