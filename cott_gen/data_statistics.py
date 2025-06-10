#!/usr/bin/env python3
"""
Data Statistics Analysis Tool for Chain-of-Thought Reasoning Data

This tool analyzes reasoning data to compute accuracy, tool usage, and other metrics.
Supports multiple output formats including text, CSV, and TSV.
"""

import os
import json
import argparse
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class FileStats:
    """Statistics for a single data file."""
    file: str
    steps: int
    video_llm: int
    vlm: int
    rag: int
    correct: bool
    na: bool


@dataclass
class OverallStats:
    """Overall statistics across all files."""
    accuracy: float
    na_rate: float
    avg_steps: float
    video_llm_ratio: float
    vlm_ratio: float
    rag_ratio: float
    total_tools: int
    total_steps: int
    total_files: int
    correct_count: int


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze chain-of-thought reasoning data statistics"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="EgoLife",
        help="Directory containing the data files"
    )
    parser.add_argument(
        "--start_idx", 
        type=int, 
        default=0,
        help="Starting index for data processing"
    )
    parser.add_argument(
        "--end_idx", 
        type=int, 
        default=None,
        help="Ending index for data processing"
    )
    parser.add_argument(
        "--output_format", 
        type=str, 
        choices=["text", "csv", "tsv"], 
        default="tsv",
        help="Output format: text (readable), csv (comma-separated), tsv (tab-separated)"
    )
    return parser.parse_args()


def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to load {file_path}: {e}")


def analyze_file_data(data: Dict[str, Any], filename: str) -> FileStats:
    """Analyze a single file's data and return statistics."""
    cot_steps = data.get("cot", [])
    
    # Count tool usage
    tool_counts = {"video_llm": 0, "vlm": 0, "rag": 0}
    for step in cot_steps:
        if "tool" in step and "name" in step["tool"]:
            tool_name = step["tool"]["name"].lower()
            for tool_type in tool_counts:
                if tool_type in tool_name:
                    tool_counts[tool_type] += 1
                    break
    
    # Determine correctness
    correct = False
    na = False
    if cot_steps:
        last_answer = cot_steps[-1].get("answer", "")
        ground_truth = data.get("ground_truth", "")
        correct = last_answer == ground_truth
        na = last_answer == "N/A"
    
    return FileStats(
        file=filename,
        steps=len(cot_steps),
        video_llm=tool_counts["video_llm"],
        vlm=tool_counts["vlm"],
        rag=tool_counts["rag"],
        correct=correct,
        na=na
    )


def compute_overall_stats(file_stats: List[FileStats]) -> OverallStats:
    """Compute overall statistics from individual file statistics."""
    total_files = len(file_stats)
    correct_count = sum(1 for stat in file_stats if stat.correct)
    na_count = sum(1 for stat in file_stats if stat.na)
    total_steps = sum(stat.steps for stat in file_stats)
    total_tools = sum(stat.video_llm + stat.vlm + stat.rag for stat in file_stats)
    
    # Avoid division by zero
    accuracy = correct_count / total_files if total_files > 0 else 0.0
    na_rate = na_count / total_files if total_files > 0 else 0.0
    avg_steps = total_steps / total_files if total_files > 0 else 0.0
    
    video_llm_total = sum(stat.video_llm for stat in file_stats)
    vlm_total = sum(stat.vlm for stat in file_stats)
    rag_total = sum(stat.rag for stat in file_stats)
    
    video_llm_ratio = video_llm_total / total_steps if total_steps > 0 else 0.0
    vlm_ratio = vlm_total / total_steps if total_steps > 0 else 0.0
    rag_ratio = rag_total / total_steps if total_steps > 0 else 0.0
    
    return OverallStats(
        accuracy=accuracy,
        na_rate=na_rate,
        avg_steps=avg_steps,
        video_llm_ratio=video_llm_ratio,
        vlm_ratio=vlm_ratio,
        rag_ratio=rag_ratio,
        total_tools=total_tools,
        total_steps=total_steps,
        total_files=total_files,
        correct_count=correct_count
    )


def print_statistics(stats: OverallStats, file_stats: List[FileStats], output_format: str) -> None:
    """Print statistics in the specified format."""
    if output_format == "text":
        _print_text_format(stats)
    else:
        _print_tabular_format(stats, file_stats, output_format)


def _print_text_format(stats: OverallStats) -> None:
    """Print statistics in human-readable text format."""
    print(f"Accuracy: {stats.accuracy:.4f}")
    print(f"NA rate: {stats.na_rate:.4f}")
    print(f"Average steps: {stats.avg_steps:.2f}")
    print(f"Video LLM: {stats.video_llm_ratio:.4f}")
    print(f"VLM: {stats.vlm_ratio:.4f}")
    print(f"RAG: {stats.rag_ratio:.4f}")
    print(f"Total tools: {stats.total_tools}")
    print(f"Total steps: {stats.total_steps}")
    print(f"Total data: {stats.total_files}")


def _print_tabular_format(stats: OverallStats, file_stats: List[FileStats], output_format: str) -> None:
    """Print statistics in CSV/TSV format."""
    sep = "," if output_format == "csv" else "\t"
    
    # Summary statistics
    print(f"Metric{sep}Value")
    print(f"Accuracy{sep}{stats.accuracy:.4f}")
    print(f"NA rate{sep}{stats.na_rate:.4f}")
    print(f"Average steps{sep}{stats.avg_steps:.2f}")
    print(f"Video LLM ratio{sep}{stats.video_llm_ratio:.4f}")
    print(f"VLM ratio{sep}{stats.vlm_ratio:.4f}")
    print(f"RAG ratio{sep}{stats.rag_ratio:.4f}")
    print(f"Total tools{sep}{stats.total_tools}")
    print(f"Total steps{sep}{stats.total_steps}")
    print(f"Total data{sep}{stats.total_files}")
    
    print()  # Blank line
    
    # Detailed file statistics
    print(f"File{sep}Steps{sep}Video_LLM{sep}VLM{sep}RAG{sep}Correct{sep}NA")
    for stat in file_stats:
        print(f"{stat.file}{sep}{stat.steps}{sep}{stat.video_llm}{sep}"
              f"{stat.vlm}{sep}{stat.rag}{sep}{int(stat.correct)}{sep}{int(stat.na)}")


def save_statistics(stats: OverallStats, file_stats: List[FileStats], 
                   data_dir: str, errors: List[str]) -> None:
    """Save statistics to CSV files."""
    save_dir = "data_statistics"
    os.makedirs(save_dir, exist_ok=True)
    
    dataset_name = os.path.basename(data_dir.rstrip('/'))
    stats_file = os.path.join(save_dir, f"data_statistics_{dataset_name}.csv")
    
    # Save error list if any
    if errors:
        error_file = os.path.join(save_dir, f"error_list_{dataset_name}.txt")
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(errors))
    
    # Save statistics
    with open(stats_file, 'w', encoding='utf-8') as f:
        # Write summary header
        f.write("Accuracy,NA rate,Average steps,Video LLM ratio,VLM ratio,RAG ratio,"
                "Total tools,Total steps,Sampled Data,Total data\n")
        f.write(f"{stats.accuracy:.4f},{stats.na_rate:.4f},{stats.avg_steps:.2f},"
                f"{stats.video_llm_ratio:.4f},{stats.vlm_ratio:.4f},{stats.rag_ratio:.4f},"
                f"{stats.total_tools},{stats.total_steps},{stats.correct_count},{stats.total_files}\n")
        
        # Write detailed statistics header
        f.write("File,Steps,Video_LLM,VLM,RAG,Correct,NA\n")
        
        # Write file statistics
        for stat in file_stats:
            f.write(f"{stat.file},{stat.steps},{stat.video_llm},{stat.vlm},"
                    f"{stat.rag},{int(stat.correct)},{int(stat.na)}\n")


def main() -> None:
    """Main function to orchestrate the data analysis."""
    args = parse_arguments()
    
    # Get sorted list of data files
    try:
        data_files = sorted(
            [f for f in os.listdir(args.data_dir) if f.endswith('.json')],
            key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf')
        )
    except FileNotFoundError:
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return
    
    # Apply index filtering
    end_idx = args.end_idx if args.end_idx is not None else len(data_files)
    selected_files = data_files[args.start_idx:end_idx]
    
    if not selected_files:
        print("No files to process in the specified range.")
        return
    
    # Process files
    file_stats = []
    errors = []
    
    for filename in selected_files:
        file_path = os.path.join(args.data_dir, filename)
        try:
            data = load_json_data(file_path)
            stats = analyze_file_data(data, filename)
            file_stats.append(stats)
        except ValueError as e:
            print(f"Warning: {e}")
            errors.append(filename)
    
    if not file_stats:
        print("No valid files were processed.")
        return
    
    # Compute and display results
    overall_stats = compute_overall_stats(file_stats)
    print_statistics(overall_stats, file_stats, args.output_format)
    save_statistics(overall_stats, file_stats, args.data_dir, errors)


if __name__ == "__main__":
    main()
