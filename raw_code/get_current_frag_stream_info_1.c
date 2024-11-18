static MOVFragmentStreamInfo * get_current_frag_stream_info(
    MOVFragmentIndex *frag_index)
{
    MOVFragmentIndexItem * item = &frag_index->item[frag_index->current];
    if (item->current >= 0 && item->current < item->nb_stream_info)
        return &item->stream_info[item->current];
    // This shouldn't happen
}