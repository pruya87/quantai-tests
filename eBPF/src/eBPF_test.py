1. Network Monitoring — Count TCP connections

from bcc import BPF

prog = """
int trace_tcp_connect(struct pt_regs *ctx) {
    bpf_trace_printk("TCP connect called\\n");
    return 0;
}
"""

b = BPF(text=prog)
b.attach_kprobe(event="tcp_v4_connect", fn_name="trace_tcp_connect")

print("Tracing TCP connect() calls... Ctrl-C to end.")
while True:
    try:
        (task, pid, cpu, flags, ts, msg) = b.trace_fields()
        print(f"{ts}: {msg}")
    except KeyboardInterrupt:
        exit()


2. Syscall Tracing — Count open() syscalls

from bcc import BPF

prog = """
int trace_open(struct pt_regs *ctx) {
    bpf_trace_printk("open syscall called\\n");
    return 0;
}
"""

b = BPF(text=prog)
b.attach_kprobe(event="sys_open", fn_name="trace_open")

print("Tracing open() syscalls... Ctrl-C to end.")
while True:
    try:
        (task, pid, cpu, flags, ts, msg) = b.trace_fields()
        print(f"{ts}: {msg}")
    except KeyboardInterrupt:
        exit()
		
3. Performance Monitoring — Track context switches

from bcc import BPF

prog = """
TRACEPOINT_PROBE(sched, sched_switch) {
    bpf_trace_printk("Context switch\\n");
    return 0;
}
"""

b = BPF(text=prog)

print("Tracing context switches... Ctrl-C to end.")
while True:
    try:
        (task, pid, cpu, flags, ts, msg) = b.trace_fields()
        print(f"{ts}: {msg}")
    except KeyboardInterrupt:
        exit()


4. Security Enforcement — Block execve for a specific process name (demo only!)

from bcc import BPF

prog = """
#include <linux/sched.h>

int kprobe__sys_execve(struct pt_regs *ctx) {
    char comm[TASK_COMM_LEN];
    bpf_get_current_comm(&comm, sizeof(comm));
    // Block execution if process name is "badproc"
    if (comm[0] == 'b' && comm[1] == 'a' && comm[2] == 'd') {
        bpf_trace_printk("Blocked execve from badproc\\n");
        return -1;  // Prevent execve
    }
    return 0;
}
"""

b = BPF(text=prog)

print("Blocking execve from process 'badproc' (demo). Ctrl-C to end.")
while True:
    try:
        (task, pid, cpu, flags, ts, msg) = b.trace_fields()
        print(f"{ts}: {msg}")
    except KeyboardInterrupt:
        exit()


5. Custom Event Logging — Trace page faults

from bcc import BPF

prog = """
TRACEPOINT_PROBE(mm, page_fault_user) {
    bpf_trace_printk("Page fault occurred\\n");
    return 0;
}
"""

b = BPF(text=prog)

print("Tracing page faults... Ctrl-C to end.")
while True:
    try:
        (task, pid, cpu, flags, ts, msg) = b.trace_fields()
        print(f"{ts}: {msg}")
    except KeyboardInterrupt:
        exit()


######################################################################
##################  DEMO: Anomaly detection agent  ###################
######################################################################

import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque
from bcc import BPF
from time import sleep

# === Anomaly detection setup ===
history_length = 10  # seconds
pid_histories = {}
model = IsolationForest(contamination=0.05)

def extract_features(counts_window):
    arr = np.array(counts_window)
    return [arr.mean(), arr.std(), arr.max()]

def update_histories_and_detect(pid_counts):
    anomalies = []
    feature_vectors = []
    pids = []

    for pid, count in pid_counts.items():
        if pid not in pid_histories:
		    # deque is used to create a rolling history of determine length
            pid_histories[pid] = deque(maxlen=history_length)
        pid_histories[pid].append(count)

        if len(pid_histories[pid]) == history_length:
            features = extract_features(pid_histories[pid])
            feature_vectors.append(features)
            pids.append(pid)

    if feature_vectors:
        X = np.array(feature_vectors)
        if not hasattr(update_histories_and_detect, "fitted"):
            model.fit(X)
            update_histories_and_detect.fitted = True
            print("Model trained on warm-up data")
            return []

        preds = model.predict(X)
        for pid, pred in zip(pids, preds):
            if pred == -1:
                anomalies.append(pid)

    return anomalies

# === eBPF program ===
prog = """
BPF_HASH(counts_openat, u32, u64);
BPF_HASH(counts_read, u32, u64);
BPF_HASH(counts_write, u32, u64);
BPF_HASH(counts_execve, u32, u64);

int trace_openat(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid();
    u64 *count = counts_openat.lookup(&pid);
    if (count) {
        (*count)++;
    } else {
        u64 init_val = 1;
        counts_openat.update(&pid, &init_val);
    }
    return 0;
}

int trace_read(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid();
    u64 *count = counts_read.lookup(&pid);
    if (count) {
        (*count)++;
    } else {
        u64 init_val = 1;
        counts_read.update(&pid, &init_val);
    }
    return 0;
}

int trace_write(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid();
    u64 *count = counts_write.lookup(&pid);
    if (count) {
        (*count)++;
    } else {
        u64 init_val = 1;
        counts_write.update(&pid, &init_val);
    }
    return 0;
}

int trace_execve(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid();
    u64 *count = counts_execve.lookup(&pid);
    if (count) {
        (*count)++;
    } else {
        u64 init_val = 1;
        counts_execve.update(&pid, &init_val);
    }
    return 0;
}

"""

# Load and attach eBPF program
b = BPF(text=prog)
b.attach_kprobe(event="sys_openat", fn_name="trace_openat")
b.attach_kprobe(event="sys_read", fn_name="trace_read")
b.attach_kprobe(event="sys_write", fn_name="trace_write")
b.attach_kprobe(event="sys_execve", fn_name="trace_execve")


print("Counting open() syscalls per process...")

try:
    while True:
        sleep(1)

        # Assuming you have these dicts from your BPF probes:
        # counts_openat = {...}, counts_read = {...}, counts_write = {...}, counts_execve = {...}

		# Extracts all the unique ks (PIDs) appearing in all list ==> {1001, 1002, 1003, 1004, 1005}
		
        all_pids = set(counts_openat) | set(counts_read) | set(counts_write) | set(counts_execve)

		# For each pid it extracts 
		# {1001: [5, 0, 4, 0], 1002: [3, 7, 0, 0], 1003: [0, 2, 0, 0], 1004: [0, 0, 1, 0], 1005: [0, 0, 0, 6]}
		
        combined_counts = {}
        for pid in all_pids:
            combined_counts[pid] = [
                counts_openat.get(pid, 0),
                counts_read.get(pid, 0),
                counts_write.get(pid, 0),
                counts_execve.get(pid, 0),
            ]

        anomalous_pids = update_histories_and_detect(combined_counts)

        for pid, counts in combined_counts.items():
            print(f"PID {pid} counts: openat={counts[0]}, read={counts[1]}, write={counts[2]}, execve={counts[3]}")

        # Clear each BPF table after reading (replace with your actual BPF table objects)
        counts_openat.clear()
        counts_read.clear()
        counts_write.clear()
        counts_execve.clear()

        if anomalous_pids:
            print("⚠️ Anomalies detected in PIDs:", anomalous_pids)

except KeyboardInterrupt:
    print("Exiting...")
