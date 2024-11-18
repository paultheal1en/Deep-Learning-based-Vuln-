static vscsi_req *vscsi_find_req(VSCSIState *s, uint32_t tag)

{

    if (tag >= VSCSI_REQ_LIMIT || !s->reqs[tag].active) {

        return NULL;

    }

    return &s->reqs[tag];

}
