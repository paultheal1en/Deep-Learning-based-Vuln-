static int multiwrite_merge(BlockDriverState *bs, BlockRequest *reqs,

    int num_reqs, MultiwriteCB *mcb)

{

    int i, outidx;



    // Sort requests by start sector

    qsort(reqs, num_reqs, sizeof(*reqs), &multiwrite_req_compare);



    // Check if adjacent requests touch the same clusters. If so, combine them,

    // filling up gaps with zero sectors.

    outidx = 0;

    for (i = 1; i < num_reqs; i++) {

        int merge = 0;

        int64_t oldreq_last = reqs[outidx].sector + reqs[outidx].nb_sectors;



        // This handles the cases that are valid for all block drivers, namely

        // exactly sequential writes and overlapping writes.

        if (reqs[i].sector <= oldreq_last) {

            merge = 1;

        }



        // The block driver may decide that it makes sense to combine requests

        // even if there is a gap of some sectors between them. In this case,

        // the gap is filled with zeros (therefore only applicable for yet

        // unused space in format like qcow2).

        if (!merge && bs->drv->bdrv_merge_requests) {

            merge = bs->drv->bdrv_merge_requests(bs, &reqs[outidx], &reqs[i]);

        }



        if (reqs[outidx].qiov->niov + reqs[i].qiov->niov + 1 > IOV_MAX) {

            merge = 0;

        }



        if (merge) {

            size_t size;

            QEMUIOVector *qiov = qemu_mallocz(sizeof(*qiov));

            qemu_iovec_init(qiov,

                reqs[outidx].qiov->niov + reqs[i].qiov->niov + 1);



            // Add the first request to the merged one. If the requests are

            // overlapping, drop the last sectors of the first request.

            size = (reqs[i].sector - reqs[outidx].sector) << 9;

            qemu_iovec_concat(qiov, reqs[outidx].qiov, size);



            // We might need to add some zeros between the two requests

            if (reqs[i].sector > oldreq_last) {

                size_t zero_bytes = (reqs[i].sector - oldreq_last) << 9;

                uint8_t *buf = qemu_blockalign(bs, zero_bytes);

                memset(buf, 0, zero_bytes);

                qemu_iovec_add(qiov, buf, zero_bytes);

                mcb->callbacks[i].free_buf = buf;

            }



            // Add the second request

            qemu_iovec_concat(qiov, reqs[i].qiov, reqs[i].qiov->size);



            reqs[outidx].nb_sectors += reqs[i].nb_sectors;

            reqs[outidx].qiov = qiov;



            mcb->callbacks[i].free_qiov = reqs[outidx].qiov;

        } else {

            outidx++;

            reqs[outidx].sector     = reqs[i].sector;

            reqs[outidx].nb_sectors = reqs[i].nb_sectors;

            reqs[outidx].qiov       = reqs[i].qiov;

        }

    }



    return outidx + 1;

}
