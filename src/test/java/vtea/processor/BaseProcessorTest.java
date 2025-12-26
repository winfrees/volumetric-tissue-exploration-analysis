/*
 * Copyright (C) 2020 Indiana University
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package vtea.processor;

import ij.ImagePlus;
import java.util.ArrayList;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import vtea.BaseTest;
import vtea.processor.listeners.ProgressListener;

/**
 * Base test class for processor tests.
 *
 * Provides utilities specific to testing SwingWorker-based processors,
 * including async execution helpers and progress monitoring.
 *
 * @author VTEA Development Team
 */
public abstract class BaseProcessorTest extends BaseTest {

    protected MockProgressListener progressListener;
    protected static final int PROCESSOR_TIMEOUT_SECONDS = 30;

    @BeforeEach
    @Override
    public void baseSetUp() {
        super.baseSetUp();
        progressListener = new MockProgressListener();
    }

    @AfterEach
    @Override
    public void baseTearDown() {
        super.baseTearDown();
        progressListener = null;
    }

    /**
     * Mock progress listener for testing processor progress updates.
     */
    protected static class MockProgressListener implements ProgressListener {

        private final ArrayList<String> messages = new ArrayList<>();
        private final ArrayList<Double> progressValues = new ArrayList<>();
        private boolean completed = false;
        private Exception error = null;

        @Override
        public void FireProgressChange(String message, double progress) {
            messages.add(message);
            progressValues.add(progress);
        }

        public void setCompleted(boolean completed) {
            this.completed = completed;
        }

        public void setError(Exception error) {
            this.error = error;
        }

        public ArrayList<String> getMessages() {
            return messages;
        }

        public ArrayList<Double> getProgressValues() {
            return progressValues;
        }

        public boolean isCompleted() {
            return completed;
        }

        public Exception getError() {
            return error;
        }

        public int getMessageCount() {
            return messages.size();
        }

        public String getLastMessage() {
            return messages.isEmpty() ? null : messages.get(messages.size() - 1);
        }

        public Double getLastProgress() {
            return progressValues.isEmpty() ? null : progressValues.get(progressValues.size() - 1);
        }

        public void reset() {
            messages.clear();
            progressValues.clear();
            completed = false;
            error = null;
        }
    }

    /**
     * Waits for a processor to complete execution.
     *
     * @param processor the processor to wait for
     * @param timeoutSeconds maximum time to wait
     * @return true if completed successfully, false if timed out
     */
    protected boolean waitForProcessor(AbstractProcessor processor, int timeoutSeconds) {
        try {
            long startTime = System.currentTimeMillis();
            long timeout = timeoutSeconds * 1000L;

            while (!processor.isDone()) {
                if (System.currentTimeMillis() - startTime > timeout) {
                    return false; // Timeout
                }
                Thread.sleep(100); // Check every 100ms
            }

            return true;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    /**
     * Executes a processor and waits for completion.
     *
     * @param processor the processor to execute
     * @throws Exception if execution fails or times out
     */
    protected void executeAndWait(AbstractProcessor processor) throws Exception {
        processor.execute();

        if (!waitForProcessor(processor, PROCESSOR_TIMEOUT_SECONDS)) {
            throw new AssertionError("Processor did not complete within " +
                    PROCESSOR_TIMEOUT_SECONDS + " seconds");
        }

        // Check for exceptions
        try {
            processor.get();
        } catch (Exception e) {
            throw new AssertionError("Processor threw exception: " + e.getMessage(), e);
        }
    }

    /**
     * Creates a mock protocol (ArrayList of processing steps).
     *
     * @return empty protocol list
     */
    protected ArrayList<Object> createMockProtocol() {
        return new ArrayList<>();
    }

    /**
     * Creates a mock image stack array for testing.
     *
     * @param width image width
     * @param height image height
     * @param depth stack depth
     * @param channels number of channels
     * @return array of ImageStacks
     */
    protected ij.ImageStack[] createMockImageStackArray(int width, int height,
                                                        int depth, int channels) {
        ij.ImageStack[] stacks = new ij.ImageStack[channels];

        for (int c = 0; c < channels; c++) {
            ij.ImageStack stack = new ij.ImageStack(width, height);
            for (int z = 0; z < depth; z++) {
                ij.process.ByteProcessor processor = new ij.process.ByteProcessor(width, height);
                // Fill with test pattern
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        processor.set(x, y, (x + y + z + c) % 256);
                    }
                }
                stack.addSlice("Slice " + (z + 1), processor);
            }
            stacks[c] = stack;
        }

        return stacks;
    }

    /**
     * Verifies that progress was reported during processor execution.
     *
     * @param minimumUpdates minimum number of progress updates expected
     * @return true if progress was reported adequately
     */
    protected boolean verifyProgressReported(int minimumUpdates) {
        return progressListener.getMessageCount() >= minimumUpdates;
    }

    /**
     * Verifies that progress values are monotonically increasing.
     *
     * @return true if progress increases or stays same
     */
    protected boolean verifyProgressMonotonic() {
        ArrayList<Double> values = progressListener.getProgressValues();
        if (values.size() < 2) return true;

        for (int i = 1; i < values.size(); i++) {
            if (values.get(i) < values.get(i - 1)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Verifies that the final progress value is 100 or close to it.
     *
     * @return true if final progress is >= 95
     */
    protected boolean verifyProgressComplete() {
        Double lastProgress = progressListener.getLastProgress();
        return lastProgress != null && lastProgress >= 95.0;
    }
}
