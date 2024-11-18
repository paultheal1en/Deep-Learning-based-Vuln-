vmxnet3_dump_rx_descr(struct Vmxnet3_RxDesc *descr)

{

    VMW_PKPRN("RX DESCR: addr %" PRIx64 ", len: %d, gen: %d, rsvd: %d, "

              "dtype: %d, ext1: %d, btype: %d",

              le64_to_cpu(descr->addr), descr->len, descr->gen,

              descr->rsvd, descr->dtype, descr->ext1, descr->btype);

}
