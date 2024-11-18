static int list_pci(ClpReqRspListPci *rrb, uint8_t *cc)

{

    S390PCIBusDevice *pbdev;

    uint32_t res_code, initial_l2, g_l2, finish;

    int rc, idx;

    uint64_t resume_token;



    rc = 0;

    if (lduw_p(&rrb->request.hdr.len) != 32) {

        res_code = CLP_RC_LEN;

        rc = -EINVAL;

        goto out;

    }



    if ((ldl_p(&rrb->request.fmt) & CLP_MASK_FMT) != 0) {

        res_code = CLP_RC_FMT;

        rc = -EINVAL;

        goto out;

    }



    if ((ldl_p(&rrb->request.fmt) & ~CLP_MASK_FMT) != 0 ||

        ldq_p(&rrb->request.reserved1) != 0 ||

        ldq_p(&rrb->request.reserved2) != 0) {

        res_code = CLP_RC_RESNOT0;

        rc = -EINVAL;

        goto out;

    }



    resume_token = ldq_p(&rrb->request.resume_token);



    if (resume_token) {

        pbdev = s390_pci_find_dev_by_idx(resume_token);

        if (!pbdev) {

            res_code = CLP_RC_LISTPCI_BADRT;

            rc = -EINVAL;

            goto out;

        }

    }



    if (lduw_p(&rrb->response.hdr.len) < 48) {

        res_code = CLP_RC_8K;

        rc = -EINVAL;

        goto out;

    }



    initial_l2 = lduw_p(&rrb->response.hdr.len);

    if ((initial_l2 - LIST_PCI_HDR_LEN) % sizeof(ClpFhListEntry)

        != 0) {

        res_code = CLP_RC_LEN;

        rc = -EINVAL;

        *cc = 3;

        goto out;

    }



    stl_p(&rrb->response.fmt, 0);

    stq_p(&rrb->response.reserved1, 0);

    stq_p(&rrb->response.reserved2, 0);

    stl_p(&rrb->response.mdd, FH_MASK_SHM);

    stw_p(&rrb->response.max_fn, PCI_MAX_FUNCTIONS);

    rrb->response.entry_size = sizeof(ClpFhListEntry);

    finish = 0;

    idx = resume_token;

    g_l2 = LIST_PCI_HDR_LEN;

    do {

        pbdev = s390_pci_find_dev_by_idx(idx);

        if (!pbdev) {

            finish = 1;

            break;

        }

        stw_p(&rrb->response.fh_list[idx - resume_token].device_id,

            pci_get_word(pbdev->pdev->config + PCI_DEVICE_ID));

        stw_p(&rrb->response.fh_list[idx - resume_token].vendor_id,

            pci_get_word(pbdev->pdev->config + PCI_VENDOR_ID));

        /* Ignore RESERVED devices. */

        stl_p(&rrb->response.fh_list[idx - resume_token].config,

            pbdev->state == ZPCI_FS_STANDBY ? 0 : 1 << 31);

        stl_p(&rrb->response.fh_list[idx - resume_token].fid, pbdev->fid);

        stl_p(&rrb->response.fh_list[idx - resume_token].fh, pbdev->fh);



        g_l2 += sizeof(ClpFhListEntry);

        /* Add endian check for DPRINTF? */

        DPRINTF("g_l2 %d vendor id 0x%x device id 0x%x fid 0x%x fh 0x%x\n",

            g_l2,

            lduw_p(&rrb->response.fh_list[idx - resume_token].vendor_id),

            lduw_p(&rrb->response.fh_list[idx - resume_token].device_id),

            ldl_p(&rrb->response.fh_list[idx - resume_token].fid),

            ldl_p(&rrb->response.fh_list[idx - resume_token].fh));

        idx++;

    } while (g_l2 < initial_l2);



    if (finish == 1) {

        resume_token = 0;

    } else {

        resume_token = idx;

    }

    stq_p(&rrb->response.resume_token, resume_token);

    stw_p(&rrb->response.hdr.len, g_l2);

    stw_p(&rrb->response.hdr.rsp, CLP_RC_OK);

out:

    if (rc) {

        DPRINTF("list pci failed rc 0x%x\n", rc);

        stw_p(&rrb->response.hdr.rsp, res_code);

    }

    return rc;

}
