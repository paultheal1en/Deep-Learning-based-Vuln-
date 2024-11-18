int trace_record_start(TraceBufferRecord *rec, TraceEventID event, size_t datasize)

{

    unsigned int idx, rec_off, old_idx, new_idx;

    uint32_t rec_len = sizeof(TraceRecord) + datasize;

    uint64_t event_u64 = event;

    uint64_t timestamp_ns = get_clock();



    do {

        old_idx = g_atomic_int_get(&trace_idx);

        smp_rmb();

        new_idx = old_idx + rec_len;



        if (new_idx - writeout_idx > TRACE_BUF_LEN) {

            /* Trace Buffer Full, Event dropped ! */

            g_atomic_int_inc(&dropped_events);

            return -ENOSPC;

        }

    } while (!g_atomic_int_compare_and_exchange(&trace_idx, old_idx, new_idx));



    idx = old_idx % TRACE_BUF_LEN;



    rec_off = idx;

    rec_off = write_to_buffer(rec_off, &event_u64, sizeof(event_u64));

    rec_off = write_to_buffer(rec_off, &timestamp_ns, sizeof(timestamp_ns));

    rec_off = write_to_buffer(rec_off, &rec_len, sizeof(rec_len));

    rec_off = write_to_buffer(rec_off, &trace_pid, sizeof(trace_pid));



    rec->tbuf_idx = idx;

    rec->rec_off  = (idx + sizeof(TraceRecord)) % TRACE_BUF_LEN;

    return 0;

}
