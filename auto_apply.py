def auto_apply(candidate_id, job_id, score):
    return {
        "candidateId": candidate_id,
        "jobId": job_id,
        "matchScore": score,
        "status": "APPLIED",
        "autoApplied": True
    }
