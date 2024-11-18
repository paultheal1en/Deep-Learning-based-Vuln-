static void mxf_free_metadataset(MXFMetadataSet **ctx, int freectx)

{

    MXFIndexTableSegment *seg;

    switch ((*ctx)->type) {

    case Descriptor:

        av_freep(&((MXFDescriptor *)*ctx)->extradata);

        break;

    case MultipleDescriptor:

        av_freep(&((MXFDescriptor *)*ctx)->sub_descriptors_refs);

        break;

    case Sequence:

        av_freep(&((MXFSequence *)*ctx)->structural_components_refs);

        break;

    case EssenceGroup:

        av_freep(&((MXFEssenceGroup *)*ctx)->structural_components_refs);

        break;

    case SourcePackage:

    case MaterialPackage:

        av_freep(&((MXFPackage *)*ctx)->tracks_refs);

        av_freep(&((MXFPackage *)*ctx)->name);


        break;

    case TaggedValue:

        av_freep(&((MXFTaggedValue *)*ctx)->name);

        av_freep(&((MXFTaggedValue *)*ctx)->value);

        break;

    case IndexTableSegment:

        seg = (MXFIndexTableSegment *)*ctx;

        av_freep(&seg->temporal_offset_entries);

        av_freep(&seg->flag_entries);

        av_freep(&seg->stream_offset_entries);

    default:

        break;

    }

    if (freectx)

    av_freep(ctx);

}