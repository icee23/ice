# Voice Conversion - Speech Quality Enhancement 🎙️

> 使用語者轉換技術於語音合成資料庫之音質改進  
> 🏆 ROCLING 2019 Poster Presentation - 2nd Place

## Overview

This project uses voice conversion technology to enhance poor-quality speech recordings in TTS corpora. By converting between different speaking rates of the same speaker, we transform voice conversion into a quality enhancement task.

## Problem & Solution

**Problem**: TTS training corpus had quality issues due to recording limitations  
**Solution**: Apply voice conversion between parallel recordings at different speaking rates

## Features

- Process 1,478 audio files across 4 speaking rates
- Support for Chinese-English mixed speech
- Multiple model architectures (GMM, DNN, DBLSTM)
- WORLD vocoder integration

## Technical Approach

### Models
- **GMM**: Traditional statistical approach
- **DNN**: 2-layer network with 2048 nodes
- **DBLSTM**: Best performance with bidirectional processing ⭐

### Key Innovation
- Added language parameters (phonetic labels + position)
- Incorporated aperiodicity conversion
- No delta features needed for DBLSTM

## Results

- **Objective**: MCD improved from 7.942 to 4.732
- **Subjective**: 70% preference for DBLSTM over DNN
- **Cross-lingual**: Successfully reduced noise in Chinese-English mixed speech


