static void channel_run(DBDMA_channel *ch)

{

    dbdma_cmd *current = &ch->current;

    uint16_t cmd, key;

    uint16_t req_count;

    uint32_t phy_addr;



    DBDMA_DPRINTF("channel_run\n");

    dump_dbdma_cmd(current);



    /* clear WAKE flag at command fetch */



    ch->regs[DBDMA_STATUS] &= cpu_to_be32(~WAKE);



    cmd = le16_to_cpu(current->command) & COMMAND_MASK;



    switch (cmd) {

    case DBDMA_NOP:

        nop(ch);

	return;



    case DBDMA_STOP:

        stop(ch);

	return;

    }



    key = le16_to_cpu(current->command) & 0x0700;

    req_count = le16_to_cpu(current->req_count);

    phy_addr = le32_to_cpu(current->phy_addr);



    if (key == KEY_STREAM4) {

        printf("command %x, invalid key 4\n", cmd);

        kill_channel(ch);

        return;

    }



    switch (cmd) {

    case OUTPUT_MORE:

        start_output(ch, key, phy_addr, req_count, 0);

	return;



    case OUTPUT_LAST:

        start_output(ch, key, phy_addr, req_count, 1);

	return;



    case INPUT_MORE:

        start_input(ch, key, phy_addr, req_count, 0);

	return;



    case INPUT_LAST:

        start_input(ch, key, phy_addr, req_count, 1);

	return;

    }



    if (key < KEY_REGS) {

        printf("command %x, invalid key %x\n", cmd, key);

        key = KEY_SYSTEM;

    }



    /* for LOAD_WORD and STORE_WORD, req_count is on 3 bits

     * and BRANCH is invalid

     */



    req_count = req_count & 0x0007;

    if (req_count & 0x4) {

        req_count = 4;

        phy_addr &= ~3;

    } else if (req_count & 0x2) {

        req_count = 2;

        phy_addr &= ~1;

    } else

        req_count = 1;



    switch (cmd) {

    case LOAD_WORD:

        load_word(ch, key, phy_addr, req_count);

	return;



    case STORE_WORD:

        store_word(ch, key, phy_addr, req_count);

	return;

    }

}
