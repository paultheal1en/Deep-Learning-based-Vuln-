static bool tracked_request_overlaps(BdrvTrackedRequest *req,

                                     int64_t offset, unsigned int bytes)

{

    /*        aaaa   bbbb */

    if (offset >= req->overlap_offset + req->overlap_bytes) {

        return false;

    }

    /* bbbb   aaaa        */

    if (req->overlap_offset >= offset + bytes) {

        return false;

    }

    return true;

}
