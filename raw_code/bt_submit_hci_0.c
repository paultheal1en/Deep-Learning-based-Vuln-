static void bt_submit_hci(struct HCIInfo *info,

                const uint8_t *data, int length)

{

    struct bt_hci_s *hci = hci_from_info(info);

    uint16_t cmd;

    int paramlen, i;



    if (length < HCI_COMMAND_HDR_SIZE)

        goto short_hci;



    memcpy(&hci->last_cmd, data, 2);



    cmd = (data[1] << 8) | data[0];

    paramlen = data[2];

    if (cmd_opcode_ogf(cmd) == 0 || cmd_opcode_ocf(cmd) == 0)	/* NOP */

        return;



    data += HCI_COMMAND_HDR_SIZE;

    length -= HCI_COMMAND_HDR_SIZE;



    if (paramlen > length)

        return;



#define PARAM(cmd, param)	(((cmd##_cp *) data)->param)

#define PARAM16(cmd, param)	le16_to_cpup(&PARAM(cmd, param))

#define PARAMHANDLE(cmd)	HNDL(PARAM(cmd, handle))

#define LENGTH_CHECK(cmd)	if (length < sizeof(cmd##_cp)) goto short_hci

    /* Note: the supported commands bitmask in bt_hci_read_local_commands_rp

     * needs to be updated every time a command is implemented here!  */

    switch (cmd) {

    case cmd_opcode_pack(OGF_LINK_CTL, OCF_INQUIRY):

        LENGTH_CHECK(inquiry);



        if (PARAM(inquiry, length) < 1) {

            bt_hci_event_complete_status(hci, HCI_INVALID_PARAMETERS);

            break;

        }



        hci->lm.inquire = 1;

        hci->lm.periodic = 0;

        hci->lm.responses_left = PARAM(inquiry, num_rsp) ?: INT_MAX;

        hci->lm.responses = 0;

        bt_hci_event_status(hci, HCI_SUCCESS);

        bt_hci_inquiry_start(hci, PARAM(inquiry, length));

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_INQUIRY_CANCEL):

        if (!hci->lm.inquire || hci->lm.periodic) {

            fprintf(stderr, "%s: Inquiry Cancel should only be issued after "

                            "the Inquiry command has been issued, a Command "

                            "Status event has been received for the Inquiry "

                            "command, and before the Inquiry Complete event "

                            "occurs", __FUNCTION__);

            bt_hci_event_complete_status(hci, HCI_COMMAND_DISALLOWED);

            break;

        }



        hci->lm.inquire = 0;

        qemu_del_timer(hci->lm.inquiry_done);

        bt_hci_event_complete_status(hci, HCI_SUCCESS);

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_PERIODIC_INQUIRY):

        LENGTH_CHECK(periodic_inquiry);



        if (!(PARAM(periodic_inquiry, length) <

                                PARAM16(periodic_inquiry, min_period) &&

                                PARAM16(periodic_inquiry, min_period) <

                                PARAM16(periodic_inquiry, max_period)) ||

                        PARAM(periodic_inquiry, length) < 1 ||

                        PARAM16(periodic_inquiry, min_period) < 2 ||

                        PARAM16(periodic_inquiry, max_period) < 3) {

            bt_hci_event_complete_status(hci, HCI_INVALID_PARAMETERS);

            break;

        }



        hci->lm.inquire = 1;

        hci->lm.periodic = 1;

        hci->lm.responses_left = PARAM(periodic_inquiry, num_rsp);

        hci->lm.responses = 0;

        hci->lm.inquiry_period = PARAM16(periodic_inquiry, max_period);

        bt_hci_event_complete_status(hci, HCI_SUCCESS);

        bt_hci_inquiry_start(hci, PARAM(periodic_inquiry, length));

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_EXIT_PERIODIC_INQUIRY):

        if (!hci->lm.inquire || !hci->lm.periodic) {

            fprintf(stderr, "%s: Inquiry Cancel should only be issued after "

                            "the Inquiry command has been issued, a Command "

                            "Status event has been received for the Inquiry "

                            "command, and before the Inquiry Complete event "

                            "occurs", __FUNCTION__);

            bt_hci_event_complete_status(hci, HCI_COMMAND_DISALLOWED);

            break;

        }

        hci->lm.inquire = 0;

        qemu_del_timer(hci->lm.inquiry_done);

        qemu_del_timer(hci->lm.inquiry_next);

        bt_hci_event_complete_status(hci, HCI_SUCCESS);

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_CREATE_CONN):

        LENGTH_CHECK(create_conn);



        if (hci->lm.connecting >= HCI_HANDLES_MAX) {

            bt_hci_event_status(hci, HCI_REJECTED_LIMITED_RESOURCES);

            break;

        }

        bt_hci_event_status(hci, HCI_SUCCESS);



        if (bt_hci_connect(hci, &PARAM(create_conn, bdaddr)))

            bt_hci_connection_reject_event(hci, &PARAM(create_conn, bdaddr));

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_DISCONNECT):

        LENGTH_CHECK(disconnect);



        if (bt_hci_handle_bad(hci, PARAMHANDLE(disconnect))) {

            bt_hci_event_status(hci, HCI_NO_CONNECTION);

            break;

        }



        bt_hci_event_status(hci, HCI_SUCCESS);

        bt_hci_disconnect(hci, PARAMHANDLE(disconnect),

                        PARAM(disconnect, reason));

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_CREATE_CONN_CANCEL):

        LENGTH_CHECK(create_conn_cancel);



        if (bt_hci_lmp_connection_ready(hci,

                                &PARAM(create_conn_cancel, bdaddr))) {

            for (i = 0; i < HCI_HANDLES_MAX; i ++)

                if (bt_hci_role_master(hci, i) && hci->lm.handle[i].link &&

                                !bacmp(&hci->lm.handle[i].link->slave->bd_addr,

                                        &PARAM(create_conn_cancel, bdaddr)))

                   break;



            bt_hci_event_complete_conn_cancel(hci, i < HCI_HANDLES_MAX ?

                            HCI_ACL_CONNECTION_EXISTS : HCI_NO_CONNECTION,

                            &PARAM(create_conn_cancel, bdaddr));

        } else

            bt_hci_event_complete_conn_cancel(hci, HCI_SUCCESS,

                            &PARAM(create_conn_cancel, bdaddr));

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_ACCEPT_CONN_REQ):

        LENGTH_CHECK(accept_conn_req);



        if (!hci->conn_req_host ||

                        bacmp(&PARAM(accept_conn_req, bdaddr),

                                &hci->conn_req_host->bd_addr)) {

            bt_hci_event_status(hci, HCI_INVALID_PARAMETERS);

            break;

        }



        bt_hci_event_status(hci, HCI_SUCCESS);

        bt_hci_connection_accept(hci, hci->conn_req_host);

        hci->conn_req_host = 0;

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_REJECT_CONN_REQ):

        LENGTH_CHECK(reject_conn_req);



        if (!hci->conn_req_host ||

                        bacmp(&PARAM(reject_conn_req, bdaddr),

                                &hci->conn_req_host->bd_addr)) {

            bt_hci_event_status(hci, HCI_INVALID_PARAMETERS);

            break;

        }



        bt_hci_event_status(hci, HCI_SUCCESS);

        bt_hci_connection_reject(hci, hci->conn_req_host,

                        PARAM(reject_conn_req, reason));

        bt_hci_connection_reject_event(hci, &hci->conn_req_host->bd_addr);

        hci->conn_req_host = 0;

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_AUTH_REQUESTED):

        LENGTH_CHECK(auth_requested);



        if (bt_hci_handle_bad(hci, PARAMHANDLE(auth_requested)))

            bt_hci_event_status(hci, HCI_NO_CONNECTION);

        else {

            bt_hci_event_status(hci, HCI_SUCCESS);

            bt_hci_event_auth_complete(hci, PARAMHANDLE(auth_requested));

        }

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_SET_CONN_ENCRYPT):

        LENGTH_CHECK(set_conn_encrypt);



        if (bt_hci_handle_bad(hci, PARAMHANDLE(set_conn_encrypt)))

            bt_hci_event_status(hci, HCI_NO_CONNECTION);

        else {

            bt_hci_event_status(hci, HCI_SUCCESS);

            bt_hci_event_encrypt_change(hci,

                            PARAMHANDLE(set_conn_encrypt),

                            PARAM(set_conn_encrypt, encrypt));

        }

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_REMOTE_NAME_REQ):

        LENGTH_CHECK(remote_name_req);



        if (bt_hci_name_req(hci, &PARAM(remote_name_req, bdaddr)))

            bt_hci_event_status(hci, HCI_NO_CONNECTION);

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_REMOTE_NAME_REQ_CANCEL):

        LENGTH_CHECK(remote_name_req_cancel);



        bt_hci_event_complete_name_cancel(hci,

                        &PARAM(remote_name_req_cancel, bdaddr));

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_READ_REMOTE_FEATURES):

        LENGTH_CHECK(read_remote_features);



        if (bt_hci_features_req(hci, PARAMHANDLE(read_remote_features)))

            bt_hci_event_status(hci, HCI_NO_CONNECTION);

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_READ_REMOTE_EXT_FEATURES):

        LENGTH_CHECK(read_remote_ext_features);



        if (bt_hci_handle_bad(hci, PARAMHANDLE(read_remote_ext_features)))

            bt_hci_event_status(hci, HCI_NO_CONNECTION);

        else {

            bt_hci_event_status(hci, HCI_SUCCESS);

            bt_hci_event_read_remote_ext_features(hci,

                            PARAMHANDLE(read_remote_ext_features));

        }

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_READ_REMOTE_VERSION):

        LENGTH_CHECK(read_remote_version);



        if (bt_hci_version_req(hci, PARAMHANDLE(read_remote_version)))

            bt_hci_event_status(hci, HCI_NO_CONNECTION);

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_READ_CLOCK_OFFSET):

        LENGTH_CHECK(read_clock_offset);



        if (bt_hci_clkoffset_req(hci, PARAMHANDLE(read_clock_offset)))

            bt_hci_event_status(hci, HCI_NO_CONNECTION);

        break;



    case cmd_opcode_pack(OGF_LINK_CTL, OCF_READ_LMP_HANDLE):

        LENGTH_CHECK(read_lmp_handle);



        /* TODO: */

        bt_hci_event_complete_lmp_handle(hci, PARAMHANDLE(read_lmp_handle));

        break;



    case cmd_opcode_pack(OGF_LINK_POLICY, OCF_HOLD_MODE):

        LENGTH_CHECK(hold_mode);



        if (PARAM16(hold_mode, min_interval) >

                        PARAM16(hold_mode, max_interval) ||

                        PARAM16(hold_mode, min_interval) < 0x0002 ||

                        PARAM16(hold_mode, max_interval) > 0xff00 ||

                        (PARAM16(hold_mode, min_interval) & 1) ||

                        (PARAM16(hold_mode, max_interval) & 1)) {

            bt_hci_event_status(hci, HCI_INVALID_PARAMETERS);

            break;

        }



        if (bt_hci_mode_change(hci, PARAMHANDLE(hold_mode),

                                PARAM16(hold_mode, max_interval),

                                acl_hold))

            bt_hci_event_status(hci, HCI_NO_CONNECTION);

        break;



    case cmd_opcode_pack(OGF_LINK_POLICY, OCF_PARK_MODE):

        LENGTH_CHECK(park_mode);



        if (PARAM16(park_mode, min_interval) >

                        PARAM16(park_mode, max_interval) ||

                        PARAM16(park_mode, min_interval) < 0x000e ||

                        (PARAM16(park_mode, min_interval) & 1) ||

                        (PARAM16(park_mode, max_interval) & 1)) {

            bt_hci_event_status(hci, HCI_INVALID_PARAMETERS);

            break;

        }



        if (bt_hci_mode_change(hci, PARAMHANDLE(park_mode),

                                PARAM16(park_mode, max_interval),

                                acl_parked))

            bt_hci_event_status(hci, HCI_NO_CONNECTION);

        break;



    case cmd_opcode_pack(OGF_LINK_POLICY, OCF_EXIT_PARK_MODE):

        LENGTH_CHECK(exit_park_mode);



        if (bt_hci_mode_cancel(hci, PARAMHANDLE(exit_park_mode),

                                acl_parked))

            bt_hci_event_status(hci, HCI_NO_CONNECTION);

        break;



    case cmd_opcode_pack(OGF_LINK_POLICY, OCF_ROLE_DISCOVERY):

        LENGTH_CHECK(role_discovery);



        if (bt_hci_handle_bad(hci, PARAMHANDLE(role_discovery)))

            bt_hci_event_complete_role_discovery(hci,

                            HCI_NO_CONNECTION, PARAMHANDLE(role_discovery), 0);

        else

            bt_hci_event_complete_role_discovery(hci,

                            HCI_SUCCESS, PARAMHANDLE(role_discovery),

                            bt_hci_role_master(hci,

                                    PARAMHANDLE(role_discovery)));

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_SET_EVENT_MASK):

        LENGTH_CHECK(set_event_mask);



        memcpy(hci->event_mask, PARAM(set_event_mask, mask), 8);

        bt_hci_event_complete_status(hci, HCI_SUCCESS);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_RESET):

        bt_hci_reset(hci);

        bt_hci_event_status(hci, HCI_SUCCESS);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_SET_EVENT_FLT):

        if (length >= 1 && PARAM(set_event_flt, flt_type) == FLT_CLEAR_ALL)

            /* No length check */;

        else

            LENGTH_CHECK(set_event_flt);



        /* Filters are not implemented */

        bt_hci_event_complete_status(hci, HCI_SUCCESS);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_FLUSH):

        LENGTH_CHECK(flush);



        if (bt_hci_handle_bad(hci, PARAMHANDLE(flush)))

            bt_hci_event_complete_flush(hci,

                            HCI_NO_CONNECTION, PARAMHANDLE(flush));

        else {

            /* TODO: ordering? */

            bt_hci_event(hci, EVT_FLUSH_OCCURRED,

                            &PARAM(flush, handle),

                            EVT_FLUSH_OCCURRED_SIZE);

            bt_hci_event_complete_flush(hci,

                            HCI_SUCCESS, PARAMHANDLE(flush));

        }

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_CHANGE_LOCAL_NAME):

        LENGTH_CHECK(change_local_name);



        if (hci->device.lmp_name)

            free((void *) hci->device.lmp_name);

        hci->device.lmp_name = strndup(PARAM(change_local_name, name),

                        sizeof(PARAM(change_local_name, name)));

        bt_hci_event_complete_status(hci, HCI_SUCCESS);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_READ_LOCAL_NAME):

        bt_hci_event_complete_read_local_name(hci);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_READ_CONN_ACCEPT_TIMEOUT):

        bt_hci_event_complete_read_conn_accept_timeout(hci);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_WRITE_CONN_ACCEPT_TIMEOUT):

        /* TODO */

        LENGTH_CHECK(write_conn_accept_timeout);



        if (PARAM16(write_conn_accept_timeout, timeout) < 0x0001 ||

                        PARAM16(write_conn_accept_timeout, timeout) > 0xb540) {

            bt_hci_event_complete_status(hci, HCI_INVALID_PARAMETERS);

            break;

        }



        hci->conn_accept_tout = PARAM16(write_conn_accept_timeout, timeout);

        bt_hci_event_complete_status(hci, HCI_SUCCESS);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_READ_SCAN_ENABLE):

        bt_hci_event_complete_read_scan_enable(hci);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_WRITE_SCAN_ENABLE):

        LENGTH_CHECK(write_scan_enable);



        /* TODO: check that the remaining bits are all 0 */

        hci->device.inquiry_scan =

                !!(PARAM(write_scan_enable, scan_enable) & SCAN_INQUIRY);

        hci->device.page_scan =

                !!(PARAM(write_scan_enable, scan_enable) & SCAN_PAGE);

        bt_hci_event_complete_status(hci, HCI_SUCCESS);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_READ_CLASS_OF_DEV):

        bt_hci_event_complete_read_local_class(hci);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_WRITE_CLASS_OF_DEV):

        LENGTH_CHECK(write_class_of_dev);



        memcpy(hci->device.class, PARAM(write_class_of_dev, dev_class),

                        sizeof(PARAM(write_class_of_dev, dev_class)));

        bt_hci_event_complete_status(hci, HCI_SUCCESS);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_READ_VOICE_SETTING):

        bt_hci_event_complete_voice_setting(hci);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_WRITE_VOICE_SETTING):

        LENGTH_CHECK(write_voice_setting);



        hci->voice_setting = PARAM(write_voice_setting, voice_setting);

        bt_hci_event_complete_status(hci, HCI_SUCCESS);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_HOST_NUMBER_OF_COMPLETED_PACKETS):

        if (length < data[0] * 2 + 1)

            goto short_hci;



        for (i = 0; i < data[0]; i ++)

            if (bt_hci_handle_bad(hci,

                                    data[i * 2 + 1] | (data[i * 2 + 2] << 8)))

                bt_hci_event_complete_status(hci, HCI_INVALID_PARAMETERS);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_READ_INQUIRY_MODE):

        /* Only if (local_features[3] & 0x40) && (local_commands[12] & 0x40)

         * else

         *     goto unknown_command */

        bt_hci_event_complete_read_inquiry_mode(hci);

        break;



    case cmd_opcode_pack(OGF_HOST_CTL, OCF_WRITE_INQUIRY_MODE):

        /* Only if (local_features[3] & 0x40) && (local_commands[12] & 0x80)

         * else

         *     goto unknown_command */

        LENGTH_CHECK(write_inquiry_mode);



        if (PARAM(write_inquiry_mode, mode) > 0x01) {

            bt_hci_event_complete_status(hci, HCI_INVALID_PARAMETERS);

            break;

        }



        hci->lm.inquiry_mode = PARAM(write_inquiry_mode, mode);

        bt_hci_event_complete_status(hci, HCI_SUCCESS);

        break;



    case cmd_opcode_pack(OGF_INFO_PARAM, OCF_READ_LOCAL_VERSION):

        bt_hci_read_local_version_rp(hci);

        break;



    case cmd_opcode_pack(OGF_INFO_PARAM, OCF_READ_LOCAL_COMMANDS):

        bt_hci_read_local_commands_rp(hci);

        break;



    case cmd_opcode_pack(OGF_INFO_PARAM, OCF_READ_LOCAL_FEATURES):

        bt_hci_read_local_features_rp(hci);

        break;



    case cmd_opcode_pack(OGF_INFO_PARAM, OCF_READ_LOCAL_EXT_FEATURES):

        LENGTH_CHECK(read_local_ext_features);



        bt_hci_read_local_ext_features_rp(hci,

                        PARAM(read_local_ext_features, page_num));

        break;



    case cmd_opcode_pack(OGF_INFO_PARAM, OCF_READ_BUFFER_SIZE):

        bt_hci_read_buffer_size_rp(hci);

        break;



    case cmd_opcode_pack(OGF_INFO_PARAM, OCF_READ_COUNTRY_CODE):

        bt_hci_read_country_code_rp(hci);

        break;



    case cmd_opcode_pack(OGF_INFO_PARAM, OCF_READ_BD_ADDR):

        bt_hci_read_bd_addr_rp(hci);

        break;



    case cmd_opcode_pack(OGF_STATUS_PARAM, OCF_READ_LINK_QUALITY):

        LENGTH_CHECK(read_link_quality);



        bt_hci_link_quality_rp(hci, PARAMHANDLE(read_link_quality));

        break;



    default:

        bt_hci_event_status(hci, HCI_UNKNOWN_COMMAND);

        break;



    short_hci:

        fprintf(stderr, "%s: HCI packet too short (%iB)\n",

                        __FUNCTION__, length);

        bt_hci_event_status(hci, HCI_INVALID_PARAMETERS);

        break;

    }

}
