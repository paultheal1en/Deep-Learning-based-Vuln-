static void build_chunks(MOVTrack *trk)

{

    int i;

    MOVIentry *chunk= &trk->cluster[0];

    uint64_t chunkSize = chunk->size;

    chunk->chunkNum= 1;

    trk->chunkCount= 1;

    for(i=1; i<trk->entry; i++){

        if(chunk->pos + chunkSize == trk->cluster[i].pos){

            chunkSize             += trk->cluster[i].size;

            chunk->samplesInChunk += trk->cluster[i].entries;

        }else{

            trk->cluster[i].chunkNum = chunk->chunkNum+1;

            chunk=&trk->cluster[i];

            chunkSize = chunk->size;

            trk->chunkCount++;

        }

    }

}
