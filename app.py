"""Suyash Voice & Memory: a local Windows desktop app."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from memory_store import DATA, ROOT, KINDS, ConflictError, Store
from speech import Recorder, Transcriber, import_segments

BG = '#F5F3EE'
INK = '#183B3B'
TEAL = '#0B7069'


class App:
    def __init__(self, root, store=None, start_worker=True):
        self.root = root
        self.store = store or Store()
        self.events = queue.Queue()
        self.recorder = None
        self.session = None
        self.correction_target = None
        self.correction_session = None
        self.memory_item = None
        self.selected_chunk = None
        self._last_transcript = None
        self.status = tk.StringVar(value='Preparing your local workspace…')
        self.project = tk.StringVar(value='Voice & Memory')
        self.capture_state = tk.StringVar(value='Ready when you are')
        self.elapsed = tk.StringVar(value='00:00')
        self.transcriber = Transcriber(self.store, self.notify)
        root.title('Suyash · Voice & Memory')
        root.geometry('1160x820')
        root.minsize(940, 720)
        root.configure(bg=BG)
        self._style()
        self._build()
        self._devices()
        self.store.recover_audio()
        self.refresh_memory()
        if start_worker:
            self.transcriber.start()
        self.root.after(150, self._poll)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.bind('<Control-space>', lambda e: self.toggle())

    def _style(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('.', font=('Segoe UI', 10), background=BG, foreground=INK)
        s.configure('TFrame', background=BG)
        s.configure('TLabel', background=BG)
        s.configure('Hero.TLabel', font=('Segoe UI Semibold', 27))
        s.configure('Sub.TLabel', foreground='#627674')
        s.configure('Title.TLabel', font=('Segoe UI Semibold', 15))
        s.configure('TButton', padding=(14, 9), background='#E8EDE7')
        s.map('TButton', background=[('active', '#D9E5DD')])
        s.configure('Accent.TButton', background=TEAL, foreground='white', font=('Segoe UI Semibold', 11))
        s.map('Accent.TButton', background=[('active', '#08534F'), ('disabled', '#A3B6B2')])
        s.configure('TNotebook', background=BG, borderwidth=0)
        s.configure('TNotebook.Tab', padding=(22, 12), background='#E8E8DF')
        s.map('TNotebook.Tab', background=[('selected', '#FFFFFF')])
        s.configure('Treeview', background='white', fieldbackground='white', rowheight=30, borderwidth=0)
        s.configure('Treeview.Heading', background='#E9EDE6', padding=8)
        s.configure('TEntry', padding=7)
        s.configure('TCombobox', padding=6)

    def _text(self, parent, height=10):
        return ScrolledText(parent, wrap='word', font=('Segoe UI', 12), height=height,
                            background='white', foreground=INK, relief='flat',
                            padx=18, pady=16, undo=True, insertbackground=TEAL)

    def _build(self):
        shell = ttk.Frame(self.root, padding=24)
        shell.pack(fill='both', expand=True)
        head = ttk.Frame(shell)
        head.pack(fill='x', pady=(0, 16))
        ttk.Label(head, text='Your thoughts, uninterrupted.', style='Hero.TLabel').pack(anchor='w')
        ttk.Label(head, text='Local voice capture. One memory library. Continue anywhere.', style='Sub.TLabel').pack(anchor='w', pady=(3, 0))
        self.tabs = ttk.Notebook(shell)
        self.tabs.pack(fill='both', expand=True)
        self.talk = ttk.Frame(self.tabs, padding=18)
        self.memory = ttk.Frame(self.tabs, padding=18)
        self.connect = ttk.Frame(self.tabs, padding=18)
        self.tabs.add(self.talk, text='Talk')
        self.tabs.add(self.memory, text='Shared memory')
        self.tabs.add(self.connect, text='Connect your AIs')
        self._talk()
        self._memory()
        self._connect()
        ttk.Label(shell, textvariable=self.status, style='Sub.TLabel', wraplength=1060).pack(anchor='w', pady=(12, 0))

    def _talk(self):
        top = ttk.Frame(self.talk)
        top.pack(fill='x')
        ttk.Label(top, text='Project').pack(side='left', padx=(0, 8))
        self.project_box = ttk.Combobox(top, textvariable=self.project, width=25,
                                      values=['Voice & Memory', 'Personal', 'YouTube'])
        self.project_box.pack(side='left')
        ttk.Label(top, text='Microphone').pack(side='left', padx=(20, 8))
        self.device = ttk.Combobox(top, state='readonly', width=34)
        self.device.pack(side='left', fill='x', expand=True)
        ttk.Button(top, text='↻', width=3, command=self._devices).pack(side='left', padx=(5, 0))
        capture = ttk.Frame(self.talk)
        capture.pack(fill='x', pady=16)
        self.start_button = ttk.Button(capture, text='●  Start talking', style='Accent.TButton', command=self.toggle)
        self.start_button.pack(side='left')
        ttk.Label(capture, textvariable=self.capture_state, style='Title.TLabel').pack(side='left', padx=20)
        self.level = ttk.Progressbar(capture, maximum=1, length=110)
        self.level.pack(side='right', padx=12)
        ttk.Label(capture, textvariable=self.elapsed, font=('Segoe UI', 18)).pack(side='right')
        ttk.Label(self.talk, text='Ctrl + Space starts or stops while this window is focused. Audio saves as you speak.', style='Sub.TLabel').pack(anchor='w', pady=(0, 8))
        self.transcript = self._text(self.talk, height=10)
        self.transcript.pack(fill='both', expand=True)
        self.transcript.configure(state='disabled')
        actions = ttk.Frame(self.talk)
        actions.pack(fill='x', pady=10)
        ttk.Button(actions, text='Copy text', command=self.copy_transcript).pack(side='left')
        ttk.Button(actions, text='Share as a note', command=self.share_note).pack(side='left', padx=7)
        ttk.Button(actions, text='Import audio', command=self.import_audio).pack(side='left')
        ttk.Button(actions, text='My vocabulary', command=self.vocabulary).pack(side='right')
        ttk.Button(actions, text='Recognition & learning', command=self.learning_settings).pack(side='right', padx=7)
        ttk.Label(self.talk, text='Improve a segment · listen, re-say, then confirm the correct words', style='Sub.TLabel').pack(anchor='w', pady=(2, 5))
        row = ttk.Frame(self.talk)
        row.pack(fill='x')
        self.segment = ttk.Combobox(row, state='readonly', width=25)
        self.segment.pack(side='left')
        self.segment.bind('<<ComboboxSelected>>', lambda e: self.choose_chunk())
        ttk.Button(row, text='Listen', command=self.listen).pack(side='left', padx=6)
        self.correct_button = ttk.Button(row, text='Re-say correction', command=self.record_correction)
        self.correct_button.pack(side='left')
        ttk.Button(row, text='Save correction', command=self.save_correction).pack(side='right')
        self.correction = self._text(self.talk, height=2)
        self.correction.pack(fill='x', pady=(7, 0))
        ttk.Label(self.talk, text='Corrections teach project vocabulary now. Review learned replacements in Recognition & learning.', style='Sub.TLabel').pack(anchor='w', pady=(5, 0))

    def _memory(self):
        bar = ttk.Frame(self.memory)
        bar.pack(fill='x', pady=(0, 12))
        self.search = tk.StringVar()
        entry = ttk.Entry(bar, textvariable=self.search)
        entry.pack(side='left', fill='x', expand=True)
        entry.bind('<Return>', lambda e: self.refresh_memory())
        ttk.Button(bar, text='Search', command=self.refresh_memory).pack(side='left', padx=7)
        ttk.Button(bar, text='Refresh', command=self.refresh_memory).pack(side='left')
        ttk.Button(bar, text='New note', command=self.new_memory).pack(side='left', padx=(7, 0))
        self.tree = ttk.Treeview(self.memory, columns=('project', 'kind', 'status'), height=6)
        self.tree.heading('#0', text='Title')
        self.tree.column('#0', width=400)
        for key, width in [('project', 180), ('kind', 110), ('status', 110)]:
            self.tree.heading(key, text=key.title())
            self.tree.column(key, width=width)
        self.tree.pack(fill='x')
        self.tree.bind('<<TreeviewSelect>>', self.select_memory)
        fields = ttk.Frame(self.memory)
        fields.pack(fill='x', pady=12)
        self.note_title = tk.StringVar()
        self.note_project = tk.StringVar(value='Voice & Memory')
        self.note_kind = tk.StringVar(value='note')
        ttk.Entry(fields, textvariable=self.note_title, width=40).pack(side='left', fill='x', expand=True)
        self.note_project_box = ttk.Combobox(fields, textvariable=self.note_project, width=22)
        self.note_project_box.pack(side='left', padx=8)
        ttk.Combobox(fields, textvariable=self.note_kind, state='readonly', values=KINDS, width=13).pack(side='left')
        self.note_body = self._text(self.memory, height=8)
        self.note_body.pack(fill='both', expand=True)
        foot = ttk.Frame(self.memory)
        foot.pack(fill='x', pady=(12, 0))
        ttk.Button(foot, text='Save privately', command=lambda: self.save_memory('private')).pack(side='left')
        ttk.Button(foot, text='Confirm & share', style='Accent.TButton', command=lambda: self.save_memory('confirmed')).pack(side='left', padx=8)
        ttk.Button(foot, text='Open recording', command=self.open_session).pack(side='left')
        ttk.Button(foot, text='Copy project handoff', command=self.copy_handoff).pack(side='right')
        ttk.Label(self.memory, text='Private = only in this app. Proposed = an AI suggestion. Confirmed = available to connected AIs.', style='Sub.TLabel').pack(anchor='w', pady=(9, 0))

    def _connect(self):
        ttk.Label(self.connect, text='One library, different ways to reach it', style='Title.TLabel').pack(anchor='w')
        info = self._text(self.connect, height=18)
        info.pack(fill='both', expand=True, pady=12)
        info.insert('1.0',
            'CLAUDE DESKTOP + CODEX\n'
            'Use the included local MCP server. It searches confirmed memories, reads their sources, '
            'and proposes new notes for you to review. Connection files are in the connections folder.\n\n'
            'CHATGPT\n'
            'A compatible desktop/local-computer setup can connect to MCP. Browser chats need a '
            'supported remote connection or an uploaded/pasted project handoff. This build does not '
            'publish your memory to the internet.\n\n'
            'GEMINI\n'
            'Gemini CLI can use the local MCP server. For the Gemini website, copy or upload a '
            'project handoff. A connection in the CLI does not automatically connect the website.\n\n'
            'MUSE (WEB APP)\n'
            'Use Copy project handoff, then paste it into Muse. Direct integration needs the exact '
            'Muse website and a supported connector. No automatic Muse sync is configured.\n\n'
            'HOW A HANDOFF WORKS\n'
            'Share the project goal, decisions, links to actual files, progress and next action. '
            'Copy that context to the next AI. Save its final handoff here to continue elsewhere. '
            'Generated files and code must also be accessible to the receiving AI.\n\n'
            'YOUR DATA\n'
            f'{self.store.root}\n\n'
            'Recordings and transcription stay on this computer. Memory returned to a cloud AI '
            'is sent to that provider as context. Native chat histories are not automatically imported. '
            'The SQLite database is authoritative; Markdown files are readable exports.\n\n'
            'Tip: ask every connected AI to search shared memory before continuing a project, '
            'and to propose a handoff when it finishes.')
        info.configure(state='disabled')
        buttons = ttk.Frame(self.connect)
        buttons.pack(fill='x')
        ttk.Button(buttons, text='Open connection files', command=lambda: os.startfile(ROOT / 'connections')).pack(side='left')
        ttk.Button(buttons, text='Open my data folder', command=lambda: os.startfile(self.store.root)).pack(side='left', padx=8)
        ttk.Button(buttons, text='Read the guide', command=lambda: os.startfile(ROOT / 'START HERE.md')).pack(side='left')
        ttk.Button(buttons, text='Retry saved audio', command=self.retry).pack(side='right')

    def notify(self, kind, value):
        self.events.put((kind, value))

    def _devices(self):
        try:
            import sounddevice as sd
            self.devices = [(None, 'System default microphone')]
            self.devices += [(i, f'{i}: {d["name"]}') for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]
            self.device['values'] = [r[1] for r in self.devices]
            self.device.current(0)
        except Exception as exc:
            self.devices = [(None, 'Microphone unavailable')]
            self.device['values'] = [self.devices[0][1]]
            self.device.current(0)
            self.status.set(f'Microphone setup: {exc}. Import audio still works.')

    def toggle(self):
        if self.recorder:
            self.stop_recording()
            return
        if self.correction_session:
            self.alert('Wait for the re-said correction to finish. Use Retry saved audio if transcription failed.')
            return
        if not self.transcriber.ready:
            self.alert(self.transcriber.failure or 'The speech model is still loading. Please wait.')
            return
        item = self.store.create(self.project.get() or 'Inbox', 'transcript',
                                 'Voice note · ' + datetime.now().strftime('%b %d, %I:%M %p'))
        self.session = item['id']
        self._last_transcript = None
        self.correction_session = None
        self.start_recording(item['id'])
        self.render_transcript()

    def start_recording(self, session):
        try:
            self.recorder = Recorder(self.store, session,
                                    device=self.devices[max(self.device.current(), 0)][0], notify=self.notify)
            self.recorder.start()
            self.start_button.configure(text='■  Stop recording')
            self.capture_state.set('Listening · saved locally')
            self.status.set('Transcription appears after a pause, or within about 25 seconds plus processing time.')
        except Exception as exc:
            self.recorder = None
            self.alert(f'Could not open the microphone: {exc}\nCheck Windows microphone permissions or import an audio file.')

    def stop_recording(self):
        recorder = self.recorder
        if not recorder:
            return
        try:
            recorder.stop()
        except Exception as exc:
            self.alert(str(exc))
            return
        self.recorder = None
        self.start_button.configure(text='●  Start talking')
        self.level['value'] = 0
        self.capture_state.set('Audio saved · finishing transcript')
        if recorder.error:
            self.alert(recorder.error)
        self.refresh_memory()

    def render_transcript(self):
        if not self.session:
            return
        item = self.store.get(self.session)
        text = item['body']
        if text != self._last_transcript:
            self.transcript.configure(state='normal')
            self.transcript.delete('1.0', 'end')
            self.transcript.insert('1.0', text)
            self.transcript.see('end')
            self.transcript.configure(state='disabled')
            self._last_transcript = text
        chunks = self.store.chunks(self.session)
        previous = self.selected_chunk
        self.chunk_rows = [r for r in chunks if r['status'] == 'done']
        labels = [f'Segment {r["sequence"]+1} · {r["duration"]:.0f}s' for r in self.chunk_rows]
        self.segment['values'] = labels
        if self.chunk_rows and previous not in [r['id'] for r in self.chunk_rows]:
            self.segment.current(len(labels) - 1)
            self.choose_chunk()
        pending = sum(r['status'] == 'pending' for r in chunks)
        failed = sum(r['status'] == 'failed' for r in chunks)
        if not self.recorder:
            self.capture_state.set(f'{pending} segments to transcribe' if pending else
                                   f'{failed} segments need retry' if failed else 'Saved on your computer')

    def choose_chunk(self):
        i = self.segment.current()
        if i < 0 or i >= len(getattr(self, 'chunk_rows', [])):
            return
        row = self.chunk_rows[i]
        self.selected_chunk = row['id']
        self.correction.delete('1.0', 'end')
        self.correction.insert('1.0', row['corrected'] if row['corrected'] is not None else
                               row['adapted'] if row.get('adapted') is not None else row['original'])
        if row.get('applied_rules') and row['applied_rules'] != '[]':
            self.status.set('Learned replacements applied. Original recognition: ' + row['original'][:180])

    def listen(self):
        i = self.segment.current()
        if i >= 0 and i < len(getattr(self, 'chunk_rows', [])):
            import winsound
            winsound.PlaySound(self.chunk_rows[i]['path'], winsound.SND_FILENAME | winsound.SND_ASYNC)

    def record_correction(self):
        if self.recorder:
            self.alert('Stop the current recording first.')
            return
        if not self.selected_chunk:
            self.alert('Select a segment to correct first.')
            return
        if not self.transcriber.ready:
            self.alert('The speech model is not ready yet.')
            return
        self.correction_target = self.selected_chunk
        item = self.store.create('Corrections', 'transcript', 'Re-said correction · ' + datetime.now().strftime('%I:%M %p'))
        self.correction_session = item['id']
        self.start_recording(item['id'])
        self.status.set('Re-say the complete corrected segment, then press Stop. Review the words and Save correction.')

    def save_correction(self):
        if not self.selected_chunk:
            self.alert('Select a completed segment first.')
            return
        try:
            self.store.correct(self.selected_chunk, self.correction.get('1.0', 'end-1c').strip())
            self.render_transcript()
            self.refresh_memory()
            self.status.set('Correction saved. Project vocabulary updated; review learned replacements in Recognition & learning.')
        except Exception as exc:
            self.alert(str(exc))

    def copy(self, text, message='Copied. Paste into whichever AI or notes app you want.'):
        if not text.strip():
            self.alert('There is no text to copy yet.')
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()
        self.status.set(message)

    def copy_transcript(self):
        self.copy(self.transcript.get('1.0', 'end-1c'))

    def share_note(self):
        if not self.session:
            self.alert('Record or import a voice note first.')
            return
        item = self.store.get(self.session)
        if not item['body'].strip():
            self.alert('Wait for the transcript to finish first.')
            return
        shared = self.store.create(item['project'], 'note', item['title'], item['body'],
                                  source=f'User-shared speech transcript {item["id"]}; may contain recognition errors',
                                  status='confirmed')
        self.refresh_memory()
        self.load_memory(shared)
        self.tabs.select(self.memory)
        self.status.set('Shared note created. Connected AIs can now find it; you can edit its title and contents here.')

    def import_audio(self):
        path = filedialog.askopenfilename(title='Choose an audio file', filetypes=[('Audio', '*.wav *.mp3 *.m4a *.ogg *.flac *.webm'), ('All files', '*.*')])
        if not path:
            return
        item = self.store.create(self.project.get() or 'Inbox', 'transcript', Path(path).stem)
        self.status.set('Importing audio into short recoverable segments…')
        def run():
            try:
                import_segments(path, self.store, item['id'])
                self.notify('imported', item['id'])
            except Exception as exc:
                self.notify('error', f'Import failed: {exc}')
        threading.Thread(target=run, daemon=True).start()

    def vocabulary(self):
        window = tk.Toplevel(self.root)
        window.title('My vocabulary')
        window.geometry('560x420')
        f = ttk.Frame(window, padding=20)
        f.pack(fill='both', expand=True)
        ttk.Label(f, text='Names and words you use', style='Title.TLabel').pack(anchor='w')
        ttk.Label(f, text='One per line. These hints guide recognition; they do not retrain the model.', wraplength=510).pack(anchor='w', pady=8)
        text = self._text(f, height=10)
        text.pack(fill='both', expand=True)
        text.insert('1.0', '\n'.join(self.store.vocabulary()))
        def save():
            try:
                for phrase in text.get('1.0', 'end').splitlines():
                    if phrase.strip():
                        self.store.add_vocabulary(phrase)
                self.status.set('Vocabulary hints saved for the next audio segment.')
                window.destroy()
            except Exception as exc:
                self.alert(str(exc))
        ttk.Button(f, text='Add these hints', command=save, style='Accent.TButton').pack(anchor='e', pady=(10, 0))

    def learning_settings(self):
        from recognition_config import installed_models, load_config, save_config
        config = load_config(self.store.root)
        window = tk.Toplevel(self.root)
        window.title('Recognition & learning')
        window.geometry('820x650')
        f = ttk.Frame(window, padding=20)
        f.pack(fill='both', expand=True)
        ttk.Label(f, text='Choose the model; keep its raw output', style='Title.TLabel').pack(anchor='w')
        ttk.Label(f, text='Settings apply after restart. Use Install Turbo.cmd for the larger model. CPU works without CUDA.', wraplength=760).pack(anchor='w', pady=8)
        row = ttk.Frame(f)
        row.pack(fill='x')
        model = tk.StringVar(value=config['model'])
        device = tk.StringVar(value=config['device'])
        ttk.Combobox(row, textvariable=model, values=installed_models(), state='readonly', width=26).pack(side='left')
        ttk.Combobox(row, textvariable=device, values=['cpu','auto','cuda'], state='readonly', width=10).pack(side='left', padx=8)
        enabled = tk.BooleanVar(value=config['personalization'])
        ttk.Checkbutton(row, text='Use personal hints and approved replacements', variable=enabled).pack(side='left')
        def save():
            config.update(model=model.get(), device=device.get(), personalization=enabled.get())
            save_config(config, self.store.root)
            self.status.set('Recognition settings saved. Restart the app to apply them.')
        ttk.Button(f, text='Save settings for next launch', command=save).pack(anchor='w', pady=10)
        project = self.store.get(self.session)['project'] if self.session else self.project.get()
        ttk.Label(f, text=f'Learned replacements · {project}', style='Title.TLabel').pack(anchor='w', pady=(10, 5))
        ttk.Label(f, text='Corrections add hints automatically. Whole-word rewriting requires your approval and can be disabled.', wraplength=760).pack(anchor='w')
        tree = ttk.Treeview(f, columns=('target','examples','enabled'), height=7)
        tree.heading('#0', text='Heard as')
        tree.column('#0', width=230)
        for key in ('target', 'examples', 'enabled'):
            tree.heading(key, text={'target':'Replace with', 'examples':'Audio examples', 'enabled':'Enabled'}[key])
            tree.column(key, width=220 if key == 'target' else 100)
        tree.pack(fill='both', expand=True, pady=8)
        def refresh():
            tree.delete(*tree.get_children())
            for r in self.store.adaptation_rules(project):
                tree.insert('', 'end', iid=str(r['id']), text=r['source'], values=(r['target'],r['examples'],'Yes' if r['enabled'] else 'No'))
        def rule(enabled):
            if tree.selection():
                self.store.set_rule(int(tree.selection()[0]), enabled)
                refresh()
        buttons = ttk.Frame(f)
        buttons.pack(fill='x')
        ttk.Button(buttons, text='Enable selected', command=lambda: rule(True)).pack(side='left')
        ttk.Button(buttons, text='Disable selected', command=lambda: rule(False)).pack(side='left', padx=8)
        def raw():
            row = next((r for r in self.store.chunks(self.session) if r['id'] == self.selected_chunk), None)
            if not row:
                self.alert('Choose an audio segment in Talk first.')
                return
            details = tk.Toplevel(window)
            details.title('Original and adjusted recognition')
            details.geometry('760x480')
            text = self._text(details, height=15)
            text.pack(fill='both', expand=True, padx=15, pady=15)
            text.insert('1.0', 'ORIGINAL RECOGNITION\n' + row['original'] + '\n\nADJUSTED OUTPUT\n' +
                        (row['adapted'] or row['original']) + '\n\nUSER LABEL\n' +
                        (row['corrected'] if row['corrected'] is not None else '(not verified)') +
                        '\n\nAPPLIED REPLACEMENTS\n' + row['applied_rules'])
            text.configure(state='disabled')
        ttk.Button(buttons, text='Inspect original recognition', command=raw).pack(side='right')
        ttk.Label(f, text='Keep a separate test session before comparing models', style='Title.TLabel').pack(anchor='w', pady=(16, 6))
        role = tk.StringVar(value='train')
        footer = ttk.Frame(f)
        footer.pack(fill='x')
        ttk.Combobox(footer, textvariable=role, state='readonly', values=['train','validation','test'], width=15).pack(side='left')
        def verify():
            try:
                self.store.verify_chunk(self.selected_chunk, role.get())
                refresh()
                self.status.set('Selected segment verified. Its whole session is assigned to ' + role.get() + '. Test/validation labels do not teach vocabulary.')
            except Exception as exc:
                self.alert(str(exc))
        ttk.Button(footer, text='Verify selected audio segment', command=verify).pack(side='left', padx=8)
        def export():
            destination = filedialog.askdirectory(title='Choose the parent folder for a private verified dataset')
            if not destination:
                return
            try:
                from personal_dataset import export_dataset
                out = Path(destination) / ('voice-dataset-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
                summary = export_dataset(self.store, out)
                self.status.set(f'Private dataset exported to {out}. Split summary: {summary}')
            except Exception as exc:
                self.alert(str(exc))
        ttk.Button(footer, text='Export verified dataset', command=export).pack(side='right')
        ttk.Label(f, text='Listen and check the words first. Shared notes do not count as verified audio. Dataset tools export only verified segments.', wraplength=760, style='Sub.TLabel').pack(anchor='w', pady=8)
        refresh()

    def refresh_memory(self):
        selected = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        for item in self.store.list(query=self.search.get()):
            self.tree.insert('', 'end', iid=item['id'], text=item['title'], values=(item['project'], item['kind'], item['status']))
        if selected and self.tree.exists(selected[0]):
            self.tree.selection_set(selected[0])
        projects = sorted(set(self.store.projects()) | {'Voice & Memory', 'Personal', 'YouTube'})
        self.project_box['values'] = projects
        self.note_project_box['values'] = projects

    def select_memory(self, event=None):
        selected = self.tree.selection()
        if selected:
            self.load_memory(self.store.get(selected[0]))

    def load_memory(self, item):
        self.memory_item = item
        self.note_title.set(item['title'])
        self.note_project.set(item['project'])
        self.note_kind.set(item['kind'])
        self.note_body.delete('1.0', 'end')
        self.note_body.insert('1.0', item['body'])

    def new_memory(self):
        self.memory_item = None
        self.tree.selection_remove(*self.tree.selection())
        self.note_title.set('')
        self.note_kind.set('note')
        self.note_project.set(self.project.get())
        self.note_body.delete('1.0', 'end')

    def save_memory(self, status):
        try:
            fields = dict(project=self.note_project.get(), title=self.note_title.get(),
                          kind=self.note_kind.get(), body=self.note_body.get('1.0', 'end-1c'), status=status)
            if self.memory_item:
                if self.memory_item['kind'] == 'transcript' and (fields['body'] != self.memory_item['body'] or fields['kind'] != 'transcript'):
                    raise ValueError('Use Open recording to correct transcript segments, or create a new note from this text.')
                item = self.store.update(self.memory_item['id'], self.memory_item['revision'], **fields)
            else:
                item = self.store.create(**fields)
            self.refresh_memory()
            self.tree.selection_set(item['id'])
            self.load_memory(item)
            self.status.set('Saved and shared with connected AIs.' if status == 'confirmed' else 'Saved privately. Connected AIs cannot read this item.')
        except (ValueError, ConflictError) as exc:
            self.alert(str(exc))

    def open_session(self):
        if self.recorder:
            self.alert('Stop the current recording before opening another.')
            return
        if not self.memory_item or self.memory_item['kind'] != 'transcript':
            self.alert('Select a transcript in the list first.')
            return
        self.session = self.memory_item['id']
        self.project.set(self.memory_item['project'])
        self.selected_chunk = None
        self._last_transcript = None
        self.render_transcript()
        self.tabs.select(self.talk)

    def copy_handoff(self):
        project = self.note_project.get() or self.project.get()
        path, text = self.store.handoff(project)
        self.copy(text, f'Project handoff copied and saved: {path.name}. Paste or upload it to ChatGPT, Gemini, Muse or Claude.')

    def retry(self):
        self.store.recover_audio()
        self.store.retry()
        self.status.set('Saved audio queued again. No recordings were uploaded.')

    def _poll(self):
        try:
            for _ in range(30):
                kind, value = self.events.get_nowait()
                if kind in ('status', 'error'):
                    self.status.set(value)
                elif kind == 'imported':
                    if not self.recorder:
                        self.session = value
                        self.selected_chunk = None
                        self._last_transcript = None
                        self.render_transcript()
                    self.refresh_memory()
                elif kind in ('chunk', 'transcribed'):
                    if value == self.session:
                        self.render_transcript()
            # Poll the correction only after capture stops and every piece completes.
        except queue.Empty:
            pass
        if self.correction_session and not self.recorder:
            chunks = self.store.chunks(self.correction_session)
            if chunks and all(r['status'] == 'done' for r in chunks):
                text = self.store.get(self.correction_session)['body']
                self.selected_chunk = self.correction_target
                self.correction.delete('1.0', 'end')
                self.correction.insert('1.0', text)
                self.correction_session = None
                self.status.set('Re-said correction ready. Review it, then Save correction.')
        if self.recorder:
            if not self.recorder.running:
                self.stop_recording()
            else:
                self.level['value'] = self.recorder.level
                seconds = int(time.monotonic() - (self.recorder.started or time.monotonic()))
                self.elapsed.set(f'{seconds // 60:02d}:{seconds % 60:02d}')
                if self.recorder.error:
                    self.status.set(self.recorder.error)
        if self.session and not self.recorder:
            self.render_transcript()
        self.root.after(200, self._poll)

    def alert(self, text):
        messagebox.showinfo('Suyash Voice & Memory', text, parent=self.root)

    def close(self):
        if self.recorder:
            self.stop_recording()
            if self.recorder:
                return
        self.transcriber.stop()
        self.root.destroy()


def wave_file(path, rate):
    import wave
    w = wave.open(str(path), 'wb')
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    return w


if __name__ == '__main__':
    # Prevent two capture apps from recovering/indexing one another's active recordings.
    import ctypes
    mutex = ctypes.windll.kernel32.CreateMutexW(None, False, 'Local\\SuyashVoiceCapture')
    if ctypes.windll.kernel32.GetLastError() == 183:
        r = tk.Tk()
        r.withdraw()
        messagebox.showinfo('Suyash Voice & Memory', 'The voice app is already open. Look for its window in the taskbar.')
    else:
        import logging
        import sys
        DATA.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(filename=DATA / 'app.log', level=logging.ERROR)
        def report(kind, value, traceback):
            logging.error('Application error', exc_info=(kind, value, traceback))
            messagebox.showerror('Suyash Voice & Memory', f'{value}\n\nDetails were saved in data/app.log.')
        sys.excepthook = report
        root = tk.Tk()
        root.report_callback_exception = report
        App(root)
        root.mainloop()
