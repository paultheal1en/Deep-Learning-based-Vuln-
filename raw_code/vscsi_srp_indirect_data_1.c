static int vscsi_srp_indirect_data(VSCSIState *s, vscsi_req *req,

                                   uint8_t *buf, uint32_t len)

{

    struct srp_direct_buf *td = &req->ind_desc->table_desc;

    struct srp_direct_buf *md = req->cur_desc;

    int rc = 0;

    uint32_t llen, total = 0;



    dprintf("VSCSI: indirect segment 0x%x bytes, td va=0x%llx len=0x%x\n",

            len, (unsigned long long)td->va, td->len);



    /* While we have data ... */

    while (len) {

        /* If we have a descriptor but it's empty, go fetch a new one */

        if (md && md->len == 0) {

            /* More local available, use one */

            if (req->local_desc) {

                md = ++req->cur_desc;

                --req->local_desc;

                --req->total_desc;

                td->va += sizeof(struct srp_direct_buf);

            } else {

                md = req->cur_desc = NULL;

            }

        }

        /* No descriptor at hand, fetch one */

        if (!md) {

            if (!req->total_desc) {

                dprintf("VSCSI:   Out of descriptors !\n");

                break;

            }

            md = req->cur_desc = &req->ext_desc;

            dprintf("VSCSI:   Reading desc from 0x%llx\n",

                    (unsigned long long)td->va);

            rc = spapr_tce_dma_read(&s->vdev, td->va, md,

                                    sizeof(struct srp_direct_buf));

            if (rc) {

                dprintf("VSCSI: tce_dma_read -> %d reading ext_desc\n", rc);

                break;

            }

            vscsi_swap_desc(md);

            td->va += sizeof(struct srp_direct_buf);

            --req->total_desc;

        }

        dprintf("VSCSI:   [desc va=0x%llx,len=0x%x] remaining=0x%x\n",

                (unsigned long long)md->va, md->len, len);



        /* Perform transfer */

        llen = MIN(len, md->len);

        if (req->writing) { /* writing = to device = reading from memory */

            rc = spapr_tce_dma_read(&s->vdev, md->va, buf, llen);

        } else {

            rc = spapr_tce_dma_write(&s->vdev, md->va, buf, llen);

        }

        if (rc) {

            dprintf("VSCSI: tce_dma_r/w(%d) -> %d\n", req->writing, rc);

            break;

        }

        dprintf("VSCSI:     data: %02x %02x %02x %02x...\n",

                buf[0], buf[1], buf[2], buf[3]);



        len -= llen;

        buf += llen;

        total += llen;

        md->va += llen;

        md->len -= llen;

    }

    return rc ? -1 : total;

}
