vmxnet3_read_next_rx_descr(VMXNET3State *s, int qidx, int ridx,

                           struct Vmxnet3_RxDesc *dbuf, uint32_t *didx)

{

    Vmxnet3Ring *ring = &s->rxq_descr[qidx].rx_ring[ridx];

    *didx = vmxnet3_ring_curr_cell_idx(ring);

    vmxnet3_ring_read_curr_cell(ring, dbuf);

}
