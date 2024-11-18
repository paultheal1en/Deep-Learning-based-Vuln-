static void send_scsi_cdb_read10(QPCIDevice *dev, void *ide_base,

                                 uint64_t lba, int nblocks)

{

    Read10CDB pkt = { .padding = 0 };

    int i;



    g_assert_cmpint(lba, <=, UINT32_MAX);

    g_assert_cmpint(nblocks, <=, UINT16_MAX);

    g_assert_cmpint(nblocks, >=, 0);



    /* Construct SCSI CDB packet */

    pkt.opcode = 0x28;

    pkt.lba = cpu_to_be32(lba);

    pkt.nblocks = cpu_to_be16(nblocks);



    /* Send Packet */

    for (i = 0; i < sizeof(Read10CDB)/2; i++) {

        qpci_io_writew(dev, ide_base + reg_data,

                       le16_to_cpu(((uint16_t *)&pkt)[i]));

    }

}
