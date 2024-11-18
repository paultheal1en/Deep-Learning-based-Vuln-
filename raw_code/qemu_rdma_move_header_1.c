static void qemu_rdma_move_header(RDMAContext *rdma, int idx,

                                  RDMAControlHeader *head)

{

    rdma->wr_data[idx].control_len = head->len;

    rdma->wr_data[idx].control_curr =

        rdma->wr_data[idx].control + sizeof(RDMAControlHeader);

}
