vmxnet3_dump_tx_descr(struct Vmxnet3_TxDesc *descr)

{

    VMW_PKPRN("TX DESCR: "

              "addr %" PRIx64 ", len: %d, gen: %d, rsvd: %d, "

              "dtype: %d, ext1: %d, msscof: %d, hlen: %d, om: %d, "

              "eop: %d, cq: %d, ext2: %d, ti: %d, tci: %d",

              le64_to_cpu(descr->addr), descr->len, descr->gen, descr->rsvd,

              descr->dtype, descr->ext1, descr->msscof, descr->hlen, descr->om,

              descr->eop, descr->cq, descr->ext2, descr->ti, descr->tci);

}
