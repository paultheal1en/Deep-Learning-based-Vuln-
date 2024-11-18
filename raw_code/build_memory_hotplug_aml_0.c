void build_memory_hotplug_aml(Aml *table, uint32_t nr_mem,

                              uint16_t io_base, uint16_t io_len,

                              const char *res_root,

                              const char *event_handler_method)

{

    int i;

    Aml *ifctx;

    Aml *method;

    Aml *sb_scope;

    Aml *mem_ctrl_dev;

    char *scan_path;

    char *mhp_res_path = g_strdup_printf("%s." MEMORY_HOTPLUG_DEVICE, res_root);



    mem_ctrl_dev = aml_device("%s", mhp_res_path);

    {

        Aml *crs;

        Aml *field;

        Aml *one = aml_int(1);

        Aml *zero = aml_int(0);

        Aml *ret_val = aml_local(0);

        Aml *slot_arg0 = aml_arg(0);

        Aml *slots_nr = aml_name(MEMORY_SLOTS_NUMBER);

        Aml *ctrl_lock = aml_name(MEMORY_SLOT_LOCK);

        Aml *slot_selector = aml_name(MEMORY_SLOT_SLECTOR);



        aml_append(mem_ctrl_dev, aml_name_decl("_HID", aml_string("PNP0A06")));

        aml_append(mem_ctrl_dev,

            aml_name_decl("_UID", aml_string("Memory hotplug resources")));



        assert(nr_mem <= ACPI_MAX_RAM_SLOTS);

        aml_append(mem_ctrl_dev,

            aml_name_decl(MEMORY_SLOTS_NUMBER, aml_int(nr_mem))

        );



        crs = aml_resource_template();

        aml_append(crs,

            aml_io(AML_DECODE16, io_base, io_base, 0, io_len)

        );

        aml_append(mem_ctrl_dev, aml_name_decl("_CRS", crs));



        aml_append(mem_ctrl_dev, aml_operation_region(

            MEMORY_HOTPLUG_IO_REGION, AML_SYSTEM_IO,

            aml_int(io_base), io_len)

        );



        field = aml_field(MEMORY_HOTPLUG_IO_REGION, AML_DWORD_ACC,

                          AML_NOLOCK, AML_PRESERVE);

        aml_append(field, /* read only */

            aml_named_field(MEMORY_SLOT_ADDR_LOW, 32));

        aml_append(field, /* read only */

            aml_named_field(MEMORY_SLOT_ADDR_HIGH, 32));

        aml_append(field, /* read only */

            aml_named_field(MEMORY_SLOT_SIZE_LOW, 32));

        aml_append(field, /* read only */

            aml_named_field(MEMORY_SLOT_SIZE_HIGH, 32));

        aml_append(field, /* read only */

            aml_named_field(MEMORY_SLOT_PROXIMITY, 32));

        aml_append(mem_ctrl_dev, field);



        field = aml_field(MEMORY_HOTPLUG_IO_REGION, AML_BYTE_ACC,

                          AML_NOLOCK, AML_WRITE_AS_ZEROS);

        aml_append(field, aml_reserved_field(160 /* bits, Offset(20) */));

        aml_append(field, /* 1 if enabled, read only */

            aml_named_field(MEMORY_SLOT_ENABLED, 1));

        aml_append(field,

            /*(read) 1 if has a insert event. (write) 1 to clear event */

            aml_named_field(MEMORY_SLOT_INSERT_EVENT, 1));

        aml_append(field,

            /* (read) 1 if has a remove event. (write) 1 to clear event */

            aml_named_field(MEMORY_SLOT_REMOVE_EVENT, 1));

        aml_append(field,

            /* initiates device eject, write only */

            aml_named_field(MEMORY_SLOT_EJECT, 1));

        aml_append(mem_ctrl_dev, field);



        field = aml_field(MEMORY_HOTPLUG_IO_REGION, AML_DWORD_ACC,

                          AML_NOLOCK, AML_PRESERVE);

        aml_append(field, /* DIMM selector, write only */

            aml_named_field(MEMORY_SLOT_SLECTOR, 32));

        aml_append(field, /* _OST event code, write only */

            aml_named_field(MEMORY_SLOT_OST_EVENT, 32));

        aml_append(field, /* _OST status code, write only */

            aml_named_field(MEMORY_SLOT_OST_STATUS, 32));

        aml_append(mem_ctrl_dev, field);



        method = aml_method("_STA", 0, AML_NOTSERIALIZED);

        ifctx = aml_if(aml_equal(slots_nr, zero));

        {

            aml_append(ifctx, aml_return(zero));

        }

        aml_append(method, ifctx);

        /* present, functioning, decoding, not shown in UI */

        aml_append(method, aml_return(aml_int(0xB)));

        aml_append(mem_ctrl_dev, method);



        aml_append(mem_ctrl_dev, aml_mutex(MEMORY_SLOT_LOCK, 0));



        method = aml_method(MEMORY_SLOT_SCAN_METHOD, 0, AML_NOTSERIALIZED);

        {

            Aml *else_ctx;

            Aml *while_ctx;

            Aml *idx = aml_local(0);

            Aml *eject_req = aml_int(3);

            Aml *dev_chk = aml_int(1);



            ifctx = aml_if(aml_equal(slots_nr, zero));

            {

                aml_append(ifctx, aml_return(zero));

            }

            aml_append(method, ifctx);



            aml_append(method, aml_store(zero, idx));

            aml_append(method, aml_acquire(ctrl_lock, 0xFFFF));

            /* build AML that:

             * loops over all slots and Notifies DIMMs with

             * Device Check or Eject Request notifications if

             * slot has corresponding status bit set and clears

             * slot status.

             */

            while_ctx = aml_while(aml_lless(idx, slots_nr));

            {

                Aml *ins_evt = aml_name(MEMORY_SLOT_INSERT_EVENT);

                Aml *rm_evt = aml_name(MEMORY_SLOT_REMOVE_EVENT);



                aml_append(while_ctx, aml_store(idx, slot_selector));

                ifctx = aml_if(aml_equal(ins_evt, one));

                {

                    aml_append(ifctx,

                               aml_call2(MEMORY_SLOT_NOTIFY_METHOD,

                                         idx, dev_chk));

                    aml_append(ifctx, aml_store(one, ins_evt));

                }

                aml_append(while_ctx, ifctx);



                else_ctx = aml_else();

                ifctx = aml_if(aml_equal(rm_evt, one));

                {

                    aml_append(ifctx,

                        aml_call2(MEMORY_SLOT_NOTIFY_METHOD,

                                  idx, eject_req));

                    aml_append(ifctx, aml_store(one, rm_evt));

                }

                aml_append(else_ctx, ifctx);

                aml_append(while_ctx, else_ctx);



                aml_append(while_ctx, aml_add(idx, one, idx));

            }

            aml_append(method, while_ctx);

            aml_append(method, aml_release(ctrl_lock));

            aml_append(method, aml_return(one));

        }

        aml_append(mem_ctrl_dev, method);



        method = aml_method(MEMORY_SLOT_STATUS_METHOD, 1, AML_NOTSERIALIZED);

        {

            Aml *slot_enabled = aml_name(MEMORY_SLOT_ENABLED);



            aml_append(method, aml_store(zero, ret_val));

            aml_append(method, aml_acquire(ctrl_lock, 0xFFFF));

            aml_append(method,

                aml_store(aml_to_integer(slot_arg0), slot_selector));



            ifctx = aml_if(aml_equal(slot_enabled, one));

            {

                aml_append(ifctx, aml_store(aml_int(0xF), ret_val));

            }

            aml_append(method, ifctx);



            aml_append(method, aml_release(ctrl_lock));

            aml_append(method, aml_return(ret_val));

        }

        aml_append(mem_ctrl_dev, method);



        method = aml_method(MEMORY_SLOT_CRS_METHOD, 1, AML_SERIALIZED);

        {

            Aml *mr64 = aml_name("MR64");

            Aml *mr32 = aml_name("MR32");

            Aml *crs_tmpl = aml_resource_template();

            Aml *minl = aml_name("MINL");

            Aml *minh = aml_name("MINH");

            Aml *maxl =  aml_name("MAXL");

            Aml *maxh =  aml_name("MAXH");

            Aml *lenl = aml_name("LENL");

            Aml *lenh = aml_name("LENH");



            aml_append(method, aml_acquire(ctrl_lock, 0xFFFF));

            aml_append(method, aml_store(aml_to_integer(slot_arg0),

                                         slot_selector));



            aml_append(crs_tmpl,

                aml_qword_memory(AML_POS_DECODE, AML_MIN_FIXED, AML_MAX_FIXED,

                                 AML_CACHEABLE, AML_READ_WRITE,

                                 0, 0x0, 0xFFFFFFFFFFFFFFFEULL, 0,

                                 0xFFFFFFFFFFFFFFFFULL));

            aml_append(method, aml_name_decl("MR64", crs_tmpl));

            aml_append(method,

                aml_create_dword_field(mr64, aml_int(14), "MINL"));

            aml_append(method,

                aml_create_dword_field(mr64, aml_int(18), "MINH"));

            aml_append(method,

                aml_create_dword_field(mr64, aml_int(38), "LENL"));

            aml_append(method,

                aml_create_dword_field(mr64, aml_int(42), "LENH"));

            aml_append(method,

                aml_create_dword_field(mr64, aml_int(22), "MAXL"));

            aml_append(method,

                aml_create_dword_field(mr64, aml_int(26), "MAXH"));



            aml_append(method,

                aml_store(aml_name(MEMORY_SLOT_ADDR_HIGH), minh));

            aml_append(method,

                aml_store(aml_name(MEMORY_SLOT_ADDR_LOW), minl));

            aml_append(method,

                aml_store(aml_name(MEMORY_SLOT_SIZE_HIGH), lenh));

            aml_append(method,

                aml_store(aml_name(MEMORY_SLOT_SIZE_LOW), lenl));



            /* 64-bit math: MAX = MIN + LEN - 1 */

            aml_append(method, aml_add(minl, lenl, maxl));

            aml_append(method, aml_add(minh, lenh, maxh));

            ifctx = aml_if(aml_lless(maxl, minl));

            {

                aml_append(ifctx, aml_add(maxh, one, maxh));

            }

            aml_append(method, ifctx);

            ifctx = aml_if(aml_lless(maxl, one));

            {

                aml_append(ifctx, aml_subtract(maxh, one, maxh));

            }

            aml_append(method, ifctx);

            aml_append(method, aml_subtract(maxl, one, maxl));



            /* return 32-bit _CRS if addr/size is in low mem */

            /* TODO: remove it since all hotplugged DIMMs are in high mem */

            ifctx = aml_if(aml_equal(maxh, zero));

            {

                crs_tmpl = aml_resource_template();

                aml_append(crs_tmpl,

                    aml_dword_memory(AML_POS_DECODE, AML_MIN_FIXED,

                                     AML_MAX_FIXED, AML_CACHEABLE,

                                     AML_READ_WRITE,

                                     0, 0x0, 0xFFFFFFFE, 0,

                                     0xFFFFFFFF));

                aml_append(ifctx, aml_name_decl("MR32", crs_tmpl));

                aml_append(ifctx,

                    aml_create_dword_field(mr32, aml_int(10), "MIN"));

                aml_append(ifctx,

                    aml_create_dword_field(mr32, aml_int(14), "MAX"));

                aml_append(ifctx,

                    aml_create_dword_field(mr32, aml_int(22), "LEN"));

                aml_append(ifctx, aml_store(minl, aml_name("MIN")));

                aml_append(ifctx, aml_store(maxl, aml_name("MAX")));

                aml_append(ifctx, aml_store(lenl, aml_name("LEN")));



                aml_append(ifctx, aml_release(ctrl_lock));

                aml_append(ifctx, aml_return(mr32));

            }

            aml_append(method, ifctx);



            aml_append(method, aml_release(ctrl_lock));

            aml_append(method, aml_return(mr64));

        }

        aml_append(mem_ctrl_dev, method);



        method = aml_method(MEMORY_SLOT_PROXIMITY_METHOD, 1,

                            AML_NOTSERIALIZED);

        {

            Aml *proximity = aml_name(MEMORY_SLOT_PROXIMITY);



            aml_append(method, aml_acquire(ctrl_lock, 0xFFFF));

            aml_append(method, aml_store(aml_to_integer(slot_arg0),

                                         slot_selector));

            aml_append(method, aml_store(proximity, ret_val));

            aml_append(method, aml_release(ctrl_lock));

            aml_append(method, aml_return(ret_val));

        }

        aml_append(mem_ctrl_dev, method);



        method = aml_method(MEMORY_SLOT_OST_METHOD, 4, AML_NOTSERIALIZED);

        {

            Aml *ost_evt = aml_name(MEMORY_SLOT_OST_EVENT);

            Aml *ost_status = aml_name(MEMORY_SLOT_OST_STATUS);



            aml_append(method, aml_acquire(ctrl_lock, 0xFFFF));

            aml_append(method, aml_store(aml_to_integer(slot_arg0),

                                         slot_selector));

            aml_append(method, aml_store(aml_arg(1), ost_evt));

            aml_append(method, aml_store(aml_arg(2), ost_status));

            aml_append(method, aml_release(ctrl_lock));

        }

        aml_append(mem_ctrl_dev, method);



        method = aml_method(MEMORY_SLOT_EJECT_METHOD, 2, AML_NOTSERIALIZED);

        {

            Aml *eject = aml_name(MEMORY_SLOT_EJECT);



            aml_append(method, aml_acquire(ctrl_lock, 0xFFFF));

            aml_append(method, aml_store(aml_to_integer(slot_arg0),

                                         slot_selector));

            aml_append(method, aml_store(one, eject));

            aml_append(method, aml_release(ctrl_lock));

        }

        aml_append(mem_ctrl_dev, method);

    }

    aml_append(table, mem_ctrl_dev);



    sb_scope = aml_scope("_SB");

    /* build memory devices */

    for (i = 0; i < nr_mem; i++) {

        Aml *dev;

        char *s;



        dev = aml_device("MP%02X", i);

        aml_append(dev, aml_name_decl("_UID", aml_string("0x%02X", i)));

        aml_append(dev, aml_name_decl("_HID", aml_eisaid("PNP0C80")));



        method = aml_method("_CRS", 0, AML_NOTSERIALIZED);

        s = g_strdup_printf("%s.%s", mhp_res_path, MEMORY_SLOT_CRS_METHOD);

        aml_append(method, aml_return(aml_call1(s, aml_name("_UID"))));

        g_free(s);

        aml_append(dev, method);



        method = aml_method("_STA", 0, AML_NOTSERIALIZED);

        s = g_strdup_printf("%s.%s", mhp_res_path, MEMORY_SLOT_STATUS_METHOD);

        aml_append(method, aml_return(aml_call1(s, aml_name("_UID"))));

        g_free(s);

        aml_append(dev, method);



        method = aml_method("_PXM", 0, AML_NOTSERIALIZED);

        s = g_strdup_printf("%s.%s", mhp_res_path,

                            MEMORY_SLOT_PROXIMITY_METHOD);

        aml_append(method, aml_return(aml_call1(s, aml_name("_UID"))));

        g_free(s);

        aml_append(dev, method);



        method = aml_method("_OST", 3, AML_NOTSERIALIZED);

        s = g_strdup_printf("%s.%s", mhp_res_path, MEMORY_SLOT_OST_METHOD);

        aml_append(method, aml_return(aml_call4(

            s, aml_name("_UID"), aml_arg(0), aml_arg(1), aml_arg(2)

        )));

        g_free(s);

        aml_append(dev, method);



        method = aml_method("_EJ0", 1, AML_NOTSERIALIZED);

        s = g_strdup_printf("%s.%s", mhp_res_path, MEMORY_SLOT_EJECT_METHOD);

        aml_append(method, aml_return(aml_call2(

                   s, aml_name("_UID"), aml_arg(0))));

        g_free(s);

        aml_append(dev, method);



        aml_append(sb_scope, dev);

    }



    /* build Method(MEMORY_SLOT_NOTIFY_METHOD, 2) {

     *     If (LEqual(Arg0, 0x00)) {Notify(MP00, Arg1)} ... }

     */

    method = aml_method(MEMORY_SLOT_NOTIFY_METHOD, 2, AML_NOTSERIALIZED);

    for (i = 0; i < nr_mem; i++) {

        ifctx = aml_if(aml_equal(aml_arg(0), aml_int(i)));

        aml_append(ifctx,

            aml_notify(aml_name("MP%.02X", i), aml_arg(1))

        );

        aml_append(method, ifctx);

    }

    aml_append(sb_scope, method);

    aml_append(table, sb_scope);



    method = aml_method(event_handler_method, 0, AML_NOTSERIALIZED);

    scan_path = g_strdup_printf("%s.%s", mhp_res_path, MEMORY_SLOT_SCAN_METHOD);

    aml_append(method, aml_call0(scan_path));

    g_free(scan_path);

    aml_append(table, method);



    g_free(mhp_res_path);

}
